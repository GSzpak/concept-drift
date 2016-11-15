from concept_drift.drift_reduction.cluster_informativeness.cluster_informativeness import \
    ClusterInformativenessCalculatorV1, ClusterInformativenessCalculatorV2

VERSION_TO_INFO_CALC = {
    1: ClusterInformativenessCalculatorV1(),
    2: ClusterInformativenessCalculatorV2()
}