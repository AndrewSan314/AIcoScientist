# Au-Ir-Rh Multimodal Epistemic Action Benchmark

**Evaluation Target**: 4 Discovery Policies on Real Au-Ir-Rh Multimodal Dataset  
**Budget Limit**: 60.0 Normalized Cost Units  
**Random Seeds**: [42, 43, 44, 45, 46]  

## Policy Comparison Summary

| Policy | Mean Best $k^0$ [cm/s] | Mean Rel Regret | Mean XRD Tests | Mean Property Tests | Mean Cost |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **fixed_ratio** | 0.011719 | 0.1748 | 14.0 | 9.0 | 59.0 |
| **property_only** | 0.011719 | 0.1748 | 2.0 | 11.0 | 57.0 |
| **random_action** | 0.011789 | 0.1698 | 2.0 | 11.0 | 57.0 |
| **scientific_action** | 0.012103 | 0.1478 | 2.2 | 11.0 | 57.2 |

*Note: Illustrative development benchmark on 5 seeds demonstrating adaptive multi-action exploration under normalized cost constraints.*
