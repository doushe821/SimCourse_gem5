
#ifndef __CPU_PRED_PERCEPTRON_HH__
#define __CPU_PRED_PERCEPTRON_HH__

#include <cstdint>
#include <vector>

#include "base/random.hh"
#include "base/types.hh"
#include "cpu/pred/conditional.hh"
#include "params/PerceptronBP.hh"

namespace gem5
{

namespace branch_prediction
{
using Weight = int16_t;
using Perceptron = std::vector<int16_t>;
using HistoryBit = int8_t;

class PerceptronBP : public ConditionalPredictor
{
  protected:
    struct BPHistory
    {
        Addr pc;
        bool prediction;
        bool condBranch;
        int output;

        std::vector<HistoryBit> historySnapshot;
    };

    const unsigned numThreads; // TODO remove
    const unsigned numPerceptrons;
    const unsigned historyLength;
    const unsigned weightBits;
    const int threshold;

    const Weight minWeight;
    const Weight maxWeight;

    std::vector<Perceptron> perceptronTable;

    std::vector<std::vector<HistoryBit>> globalHistory;

    unsigned getIndex(Addr pc) const;

    int computeOutput(const Perceptron &perceptron,
                      const std::vector<HistoryBit> &history) const;

    bool predictFromOutput(int output) const;

    bool shouldTrain(bool prediction, bool actual, int output) const;

    void train(Perceptron &perceptron, const std::vector<HistoryBit> &history,
               bool taken);

    void updateGlobalHistory(ThreadID tid, bool taken);

    void restoreGlobalHistory(ThreadID tid,
                              const std::vector<HistoryBit> &history);

    HistoryBit encodeOutcome(bool taken) const;

    Weight satChange(Weight value1, Weight value2) const;

  public:
    PerceptronBP(const PerceptronBPParams &params);
    bool lookup(ThreadID tid, Addr pc, void *&bp_history) override;

    void updateHistories(ThreadID tid, Addr pc, bool uncond, bool taken,
                         Addr target, const StaticInstPtr &inst,
                         void *&bp_history) override;
    void update(ThreadID tid, Addr pc, bool taken, void *&bp_history,
                bool squashed, const StaticInstPtr &inst,
                Addr target) override;
    void squash(ThreadID tid, void *&bp_history) override;
    void branchPlaceholder(ThreadID tid, Addr pc, bool uncond,
                           void *&bp_history) override;
};

} // namespace branch_prediction

} // namespace gem5

#endif
