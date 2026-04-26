
#include "cpu/pred/perceptron.hh"

#include <cstdlib>
#include <ranges>

#include "base/intmath.hh"
#include "base/logging.hh"
#include "base/random.hh"
#include "base/trace.hh"
#include "base/types.hh"
#include "cpu/pred/conditional.hh"
#include "debug/Fetch.hh"

namespace gem5
{

namespace branch_prediction
{

PerceptronBP::PerceptronBP(const PerceptronBPParams &params)
    : ConditionalPredictor(params),
      numThreads(params.numThreads),
      numPerceptrons(params.numPerceptrons),
      historyLength(params.historyLength),
      weightBits(params.weightBits),
      threshold(params.threshold),
      minWeight(static_cast<Weight>(-(1 << (weightBits - 1)))),
      maxWeight(static_cast<Weight>((1 << (weightBits - 1)) - 1)),
      perceptronTable(numPerceptrons, Perceptron(historyLength + 1, 0)),
      globalHistory(params.numThreads,
                    std::vector<HistoryBit>(historyLength, -1))
{
    fatal_if(numPerceptrons == 0, "PerceptronBP requires numPerceptrons > 0");
    fatal_if(historyLength == 0, "PerceptronBP requires historyLength > 0");
    fatal_if(weightBits < 2, "PerceptronBP requires weightBits >= 2");
    fatal_if(weightBits >= 15,
             "PerceptronBP currently expects small int16_t weights");
}

int
PerceptronBP::computeOutput(const Perceptron &perceptron,
                            const std::vector<HistoryBit> &history) const
{
    int output{0};
    assert(perceptron.size() == historyLength + 1);
    size_t min_length = std::min(perceptron.size() - 1, history.size());
    output += perceptron[0];
    // should probably make this std::accumulate
    for (unsigned i = 0; i < min_length; ++i) {
        output += perceptron[i + 1] * history[i];
    }

    return output;
}

bool
PerceptronBP::lookup(ThreadID tid, Addr pc, void *&bp_history)
{
    unsigned perceptron_index = getIndex(pc);
    const Perceptron &perceptron = perceptronTable[perceptron_index];
    int dotProduct = computeOutput(perceptron, globalHistory[tid]);

    bool prediction = predictFromOutput(dotProduct);

    auto *history = new BPHistory;

    history->pc = pc;
    history->prediction = prediction;
    history->condBranch = true;
    history->output = dotProduct;
    history->historySnapshot = globalHistory[tid];

    bp_history = history;
    return prediction;
}

void
PerceptronBP::update(ThreadID tid, Addr pc, bool taken, void *&bp_history,
                     bool squashed, const StaticInstPtr &inst, Addr target)
{
    if (bp_history == nullptr) {
        return;
    }

    BPHistory *history = static_cast<BPHistory *>(bp_history);

    if (squashed) {
        restoreGlobalHistory(tid, history->historySnapshot);
        updateGlobalHistory(tid, taken);

        delete history;
        bp_history = nullptr;
        return;
    }

    if (!history->condBranch) {
        delete history;
        bp_history = nullptr;
        return;
    }

    Perceptron &perceptron = perceptronTable[getIndex(pc)];

    if (shouldTrain(history->prediction, taken, history->output)) {
        train(perceptron, history->historySnapshot, taken);
    }

    delete history;
    bp_history = nullptr;
}

Weight
PerceptronBP::satChange(Weight value1, Weight value2) const
{
    int val = static_cast<int>(value1) + static_cast<int>(value2);
    if (val >= maxWeight) {
        return maxWeight;
    } else if (val <= minWeight) {
        return minWeight;
    } else {
        return val;
    }
}

void
PerceptronBP::train(Perceptron &perceptron,
                    const std::vector<HistoryBit> &history, bool taken)
{
    int16_t outcome = encodeOutcome(taken);
    perceptron[0] = satChange(perceptron[0], outcome);
    assert(perceptron.size() == historyLength + 1);
    size_t min_size = std::min(perceptron.size() - 1, history.size());
    for (unsigned i = 0; i < min_size; ++i) {
        perceptron[i + 1] = satChange(perceptron[i + 1], history[i] * outcome);
    }
}

bool
PerceptronBP::predictFromOutput(int output) const
{
    return (output >= 0);
}

unsigned
PerceptronBP::getIndex(Addr pc) const
{
    return (pc >> instShiftAmt) % numPerceptrons;
}

// TODO:
void
PerceptronBP::updateHistories(ThreadID tid, Addr pc, bool uncond, bool taken,
                              Addr target, const StaticInstPtr &inst,
                              void *&bp_history)
{
    if (bp_history == nullptr) {
        auto *history = new BPHistory;
        history->pc = pc;
        history->prediction = taken;
        history->condBranch = !uncond;
        history->output = 0;
        history->historySnapshot = globalHistory[tid];
        bp_history = history;
    }

    updateGlobalHistory(tid, taken);
}

void
PerceptronBP::updateGlobalHistory(ThreadID tid, bool taken)
{
    auto &history = globalHistory[tid];

    if (history.empty()) {
        return;
    }

    for (int i = history.size() - 1; i > 0; --i) {
        history[i] = history[i - 1];
    }

    history[0] = encodeOutcome(taken);
}

void
PerceptronBP::squash(ThreadID tid, void *&bp_history)
{
    if (bp_history == nullptr) {
        return;
    }

    BPHistory *history = static_cast<BPHistory *>(bp_history);

    restoreGlobalHistory(tid, history->historySnapshot);

    delete history;
    bp_history = nullptr;
}

void
PerceptronBP::restoreGlobalHistory(ThreadID tid,
                                   const std::vector<HistoryBit> &history)
{
    globalHistory[tid] = history;
}

HistoryBit
PerceptronBP::encodeOutcome(bool taken) const
{
    return taken ? 1 : -1;
}

bool
PerceptronBP::shouldTrain(bool prediction, bool actual, int output) const
{
    return prediction != actual || std::abs(output) <= threshold;
}

void
PerceptronBP::branchPlaceholder(ThreadID tid, Addr pc, bool uncond,
                                void *&bp_history)
{
    assert(bp_history == nullptr);

    auto *history = new BPHistory;
    history->pc = pc;
    history->prediction = false;
    history->condBranch = !uncond;
    history->output = 0;
    history->historySnapshot = globalHistory[tid];

    bp_history = history;
}

} // namespace branch_prediction

} // namespace gem5
