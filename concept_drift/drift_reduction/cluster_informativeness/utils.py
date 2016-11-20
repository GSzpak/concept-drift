from concept_drift.drift_reduction.cluster_informativeness.cluster_informativeness import \
    ClusterInformativenessCalculatorV1, ClusterInformativenessCalculatorV2, ClusterInformativenessCalculatorV3

VERSION_TO_INFO_CALC = {
    1: ClusterInformativenessCalculatorV1(),
    2: ClusterInformativenessCalculatorV2(),
    3: ClusterInformativenessCalculatorV3()
}