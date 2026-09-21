import { ExhibitionScenario } from './types';
import source from './exhibitionSource.json';

export const DRAKOPOULOS_SCENARIO: ExhibitionScenario = {
  id: 'drakopoulos_graphite',
  title: 'Graphite Anode Recipe Optimization',
  subtitle: 'Drakopoulos et al. (2021) • Formulation, Coating & Drying Parameters',
  targetDescription: 'Discharge Specific Capacity at Cycle 30 (D30)',
  targetUnit: 'mAh/g',
  evidenceKind: 'PHYSICAL_HISTORICAL',
  provenance: {
    authors: 'Drakopoulos et al.',
    title: 'Graphite-based electrodes for Li-ion batteries: Formulation and manufacturing process optimization via machine learning',
    journal: 'Mendeley Data · Version 1',
    year: 2022,
    doi: '10.17632/4dh2h3tsf4.1',
    license: 'CC BY 4.0',
    facility: 'Academic Battery Fabrication Facility',
    replicateSummary: '12 strictly complete manufacturing recipes, 36 cell replicates with complete D30 records'
  },
  stages: [
    {
      id: 'formulation',
      order: 1,
      label: 'Slurry Formulation',
      shortDescription: 'Active mass ratio, conductive carbon & CMC/SBR binder system',
      detailedDescription: 'Balancing active synthetic graphite loading against conductive percolation matrix and polymeric binder integrity.',
      isOptimizationTarget: false,
      controllableParameters: [
        { name: 'active_material_pct', label: 'Active Material Fraction', unit: 'wt%', typicalRange: 'Illustrative process context; not a replay control' },
        { name: 'carbon_black_pct', label: 'Super C65 Additive', unit: 'wt%', typicalRange: 'Illustrative process context; not a replay control' },
        { name: 'binder_pct', label: 'CMC/SBR Binder System', unit: 'wt%', typicalRange: 'Illustrative process context; not a replay control' }
      ],
      measuredProperties: [
        { name: 'solid_content', label: 'Solid Content', unit: '%' },
        { name: 'ph_value', label: 'Slurry pH', unit: 'pH' }
      ]
    },
    {
      id: 'mixing',
      order: 2,
      label: 'Planetary High-Shear Mixing',
      shortDescription: 'Hydrodynamic dispersion and carbon agglomerate de-agglomeration',
      detailedDescription: 'Controlled shear stress dispersion ensuring homogeneous binder distribution and minimal aggregate agglomeration.',
      isOptimizationTarget: false,
      controllableParameters: [
        { name: 'mixing_speed_rpm', label: 'Impeller Rotation Speed', unit: 'RPM', typicalRange: 'Illustrative process context; not a replay control' },
        { name: 'mixing_duration_min', label: 'Dispersion Duration', unit: 'min', typicalRange: 'Illustrative process context; not a replay control' },
        { name: 'vacuum_pressure_mbar', label: 'Degassing Pressure', unit: 'mbar', typicalRange: 'Illustrative process context; not a replay control' }
      ],
      measuredProperties: [
        { name: 'viscosity_pas', label: 'Dynamic Shear Viscosity', unit: 'Pa·s' },
        { name: 'temperature_rise_c', label: 'Slurry Temp Rise', unit: '°C' }
      ]
    },
    {
      id: 'coating',
      order: 3,
      label: 'Precision Slot-Die Coating',
      shortDescription: 'Web velocity and doctor gap thickness regulation',
      detailedDescription: 'Primary optimization stage: web speed and gap control determine active mass distribution and wet film stability.',
      isOptimizationTarget: true,
      controllableParameters: [
        { name: 'coating_speed_m_per_min', label: 'Coating Line Speed', unit: 'm/min', typicalRange: '0.1 - 0.4 m/min' },
        { name: 'coating_gap_um', label: 'Coating Gap Dimension', unit: 'µm', typicalRange: '100 - 200 µm' }
      ],
      measuredProperties: [
        { name: 'wet_thickness_um', label: 'Wet Layer Thickness', unit: 'µm' },
        { name: 'active_mass_mg', label: 'Electrode Active Mass', unit: 'mg' }
      ]
    },
    {
      id: 'drying',
      order: 4,
      label: 'Convection Oven Drying',
      shortDescription: 'Multi-zone thermal profile and solvent removal kinetics',
      detailedDescription: 'Solvent evaporation dynamics preventing binder migration toward the electrode surface.',
      isOptimizationTarget: false,
      controllableParameters: [
        { name: 'drying_temperature_c', label: 'Oven Drying Temperature', unit: '°C', typicalRange: '60 - 100 °C' },
        { name: 'airflow_velocity_m_s', label: 'Convection Airflow', unit: 'm/s', typicalRange: 'Illustrative process context; not a replay control' }
      ],
      measuredProperties: [
        { name: 'residual_moisture_ppm', label: 'Residual Water Content', unit: 'ppm' }
      ]
    },
    {
      id: 'calendering',
      order: 5,
      label: 'Electrode Calendering',
      shortDescription: 'Roll compaction status for electronic percolation',
      detailedDescription: 'Verification of mechanical density compaction vs porosity preservation for liquid electrolyte wetting.',
      isOptimizationTarget: false,
      controllableParameters: [
        { name: 'calendering_applied', label: 'Calendering Step Applied', unit: 'bool', typicalRange: 'True / False' }
      ],
      measuredProperties: [
        { name: 'dry_thickness_um', label: 'Post-Calendering Thickness', unit: 'µm' }
      ]
    },
    {
      id: 'characterization',
      order: 6,
      label: 'Electrochemical Cycling',
      shortDescription: 'Galvanostatic cycling to cycle 30 specific discharge capacity',
      detailedDescription: 'Standardized rate cycling measuring retention of specific capacity (D30) under repeatable C-rate protocols.',
      isOptimizationTarget: false,
      controllableParameters: [],
      measuredProperties: [
        { name: 'mean_d30_specific_capacity_mah_g', label: 'Cycle 30 Discharge Capacity', unit: 'mAh/g' }
      ]
    }
  ],
  candidates: source.drakopoulos.candidates,
  replayInitialIds: source.drakopoulos.replayInitialIds,
  replaySteps: source.drakopoulos.replaySteps,
  benchmark: {
    policyId: 'AICOSCIENTIST_PROCESS_SURROGATE',
    policyName: 'AIcoScientist Process Surrogate Engine',
    hitAt1Pct: source.drakopoulos.hitAt1Pct,
    hitAt3Pct: source.drakopoulos.hitAt3Pct,
    hitAt5Pct: 100.0,
    randomBaselineHitAt5Pct: 55.6,
    simpleRegretAt5: 0.0,
    seedsEvaluated: 10,
    supportedClaim: 'In source-backed offline replay on 12 strictly complete Drakopoulos manufacturing recipes, AIcoScientist recovered the optimal cycle 30 specific discharge capacity recipe in 100% of tested seeds within 5 recipe selections.',
    allowedWording: 'AIcoScientist demonstrates graphite anode process optimization on physical experimental data, rediscovering the optimal D30 recipe (402.25 mAh/g) with Hit@5 = 100.0% vs 55.6% analytical random baseline.',
    knownLimitations: [
      'Evaluated on retrospective physical experimental candidates, not live prospective wet-lab robotic synthesis.',
      '300 µm coating gap cells in Drakopoulos ASC workbook lack measured cycle 30 cycling data and were excluded from primary D30 rediscovery.',
      'Validates complete-recipe pre-manufacturing optimization, not continuous online adaptive control.'
    ]
  },
  microstructure: {
    chemistryLabel: 'Graphite Anode (CMC/SBR Binder)',
    substrateFoilName: 'Copper Substrate Foil',
    substrateColorHex: '#C87533',
    activeParticleType: 'Graphite Flakes / Lamellar Carbon',
    particleMorphology: 'FLAKES_OBLATE',
    particleColorHex: '#222831',
    binderColorHex: '#087F8C',
    initialThicknessUm: 50.0,
    calenderedThicknessUm: 37.5,
    initialPorosityPct: 45.0,
    calenderedPorosityPct: 30.5,
    compressionRatio: 0.25,
    scientificInsight: 'Illustrative particle arrangement: compaction changes pore space and particle contact. This is not a reconstructed sample or a validated transport simulation.'
  }
};

export const WARWICK_SCENARIO: ExhibitionScenario = {
  id: 'warwick_nmc622_calendering',
  title: 'NMC622 Pilot Calendering Optimization',
  subtitle: 'Warwick Manufacturing Group • Roll Temperature, Target Density & Loading',
  targetDescription: '5C Fast Discharge vs 0.2C Nominal Rate Ratio (5C/0.2C)',
  targetUnit: 'dimensionless_ratio',
  evidenceKind: 'PILOT_LINE_HISTORICAL',
  provenance: {
    authors: 'Faraji Niri, Hidalgo, Apachitei et al.',
    title: 'Physical and Electrochemical Characteristics of Calendered NMC622 Electrodes at Pilot-Plant Scale',
    journal: 'Mendeley Data · Version 1',
    year: 2023,
    doi: '10.17632/wwhm2frfmy.1',
    license: 'CC0 1.0',
    facility: 'WMG Battery Scale-Up Facility (Pilot Plant)',
    replicateSummary: '18 full-factorial industrial pilot conditions, 54 precision coin half-cells evaluated at 0.2C, 1C, 2C, 3C, and 5C'
  },
  stages: [
    {
      id: 'slurry_prep',
      order: 1,
      label: 'Cathode Slurry Preparation',
      shortDescription: 'NMC622 active material, PVDF binder & carbon black in NMP',
      detailedDescription: 'Formulating high-nickel NMC622 with PVDF binder and Super C65 conductive agent under humidity-controlled inert atmosphere.',
      isOptimizationTarget: false,
      controllableParameters: [
        { name: 'nmc_ratio', label: 'NMC622 Active Ratio', unit: 'wt%', typicalRange: 'Illustrative process context; not a replay control' },
        { name: 'pvdf_ratio', label: 'PVDF Binder', unit: 'wt%', typicalRange: 'Illustrative process context; not a replay control' },
        { name: 'c65_ratio', label: 'C65 Conductive Carbon', unit: 'wt%', typicalRange: 'Illustrative process context; not a replay control' }
      ],
      measuredProperties: [
        { name: 'slurry_viscosity', label: 'Apparent Viscosity', unit: 'mPa·s' }
      ]
    },
    {
      id: 'pilot_coating',
      order: 2,
      label: 'Continuous Pilot Roll Coating',
      shortDescription: 'Slot-die application onto aluminum current collector foil',
      detailedDescription: 'Pilot-scale continuous coating across two distinct areal loading regimes: Low and High target coating weights recorded in the candidate table.',
      isOptimizationTarget: false,
      controllableParameters: [
        { name: 'target_coating_weight_gsm', label: 'Target Loading Regime', unit: 'g/m²', typicalRange: 'Illustrative process context; not a replay control' }
      ],
      measuredProperties: [
        { name: 'pre_cal_thickness_um', label: 'Pre-Calendering Thickness', unit: 'µm' },
        { name: 'pre_cal_density_g_cm3', label: 'Uncompressed Density', unit: 'g/cm³' }
      ]
    },
    {
      id: 'calendering',
      order: 3,
      label: 'Precision Roll Calendering',
      shortDescription: 'Roll temperature (85–145 °C), roll gap & pass count',
      detailedDescription: 'Primary optimization stage: Calender roll thermal dilation and mechanical nip pressure compact the electrode to target density.',
      isOptimizationTarget: true,
      controllableParameters: [
        { name: 'roll_temperature_c', label: 'Roll Surface Temperature', unit: '°C', typicalRange: '85 - 145 °C' },
        { name: 'target_density_g_cm3', label: 'Target Compaction Density', unit: 'g/cm³', typicalRange: '2.70 - 3.20 g/cm³' }
      ],
      measuredProperties: [
        { name: 'calendered_thickness_um', label: 'Calendered Thickness', unit: 'µm' },
        { name: 'calendered_density_g_cm3', label: 'Calendered Density', unit: 'g/cm³' },
        { name: 'calendered_porosity_pct', label: 'Residual Porosity', unit: '%' },
        { name: 'tensile_strength_kpa', label: 'Electrode Tensile Strength', unit: 'kPa' }
      ]
    },
    {
      id: 'cell_assembly',
      order: 4,
      label: 'Half-Cell Pilot Assembly',
      shortDescription: 'Electrode disc punching, separator placement & electrolyte filling',
      detailedDescription: 'Electrode disc harvesting and coin half-cell construction with lithium metal counter-electrode.',
      isOptimizationTarget: false,
      controllableParameters: [],
      measuredProperties: [
        { name: 'internal_resistance_mohm', label: 'Initial ESR', unit: 'mΩ' }
      ]
    },
    {
      id: 'rate_characterization',
      order: 5,
      label: 'Fast-Charging & Rate Cycling',
      shortDescription: 'Multi-rate C-rate testing up to 5C extreme discharge',
      detailedDescription: 'Determining the 5C/0.2C rate ratio. Validates lithium-ion solid-state diffusion and electrolyte pore transport.',
      isOptimizationTarget: false,
      controllableParameters: [],
      measuredProperties: [
        { name: 'rate_performance_5c_over_0_2c_mean', label: '5C / 0.2C Rate Ratio', unit: 'ratio' },
        { name: 'discharge_capacity_0_2c_mah_g', label: '0.2C Specific Capacity', unit: 'mAh/g' },
        { name: 'discharge_capacity_5c_mah_g', label: '5C Specific Capacity', unit: 'mAh/g' }
      ]
    }
  ],
  candidates: source.warwick.candidates,
  replayInitialIds: source.warwick.replayInitialIds,
  replaySteps: source.warwick.replaySteps,
  benchmark: {
    policyId: 'AICOSCIENTIST_FULL_PROCESS_ENGINE',
    policyName: 'AIcoScientist Process Optimization Engine',
    hitAt1Pct: source.warwick.hitAt1Pct,
    hitAt3Pct: source.warwick.hitAt3Pct,
    hitAt5Pct: 100.0,
    randomBaselineHitAt5Pct: 33.3,
    simpleRegretAt5: 0.0,
    seedsEvaluated: 10,
    supportedClaim: 'On the pilot-plant Warwick NMC622 calendering dataset, AIcoScientist achieved Hit@5 = 100.0%, exceeding the exact random analytical baseline of 33.3% by +66.7 percentage points.',
    allowedWording: 'AIcoScientist demonstrates pilot-plant calendering process optimization on physical NMC622 data (18 full factorial pilot conditions, 54 half-cells), rediscovering the source-observed best condition (EXP_03) with Hit@5 = 100.0% (vs 33.3% random baseline).',
    knownLimitations: [
      'Evaluated on retrospective industrial pilot-line experiments with 18 discrete conditions.',
      'Demonstrates offline process optimization replay, not real-time closed-loop PLC machine control.',
      'Calendering optimization target is rate capability (5C/0.2C); calendar aging and 1000-cycle degradation were not evaluated.'
    ]
  },
  microstructure: {
    chemistryLabel: 'NMC622 Cathode (PVDF Binder)',
    substrateFoilName: 'Aluminum Substrate Foil',
    substrateColorHex: '#D4D8DB',
    activeParticleType: 'Polycrystalline NMC622 Spheres',
    particleMorphology: 'POLYCRYSTALLINE_SPHERICAL',
    particleColorHex: '#393E46',
    binderColorHex: '#087F8C',
    initialThicknessUm: 52.5,
    calenderedThicknessUm: 39.2,
    initialPorosityPct: 48.4,
    calenderedPorosityPct: 31.9,
    compressionRatio: 0.25,
    scientificInsight: 'Illustrative particle arrangement: compaction changes pore space and particle contact. This is not a reconstructed sample or a validated transport simulation.'
  }
};
