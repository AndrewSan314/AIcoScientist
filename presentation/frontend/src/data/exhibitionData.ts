import { ExhibitionScenario } from './types';

export const DRAKOPOULOS_SCENARIO: ExhibitionScenario = {
  id: 'drakopoulos_graphite',
  title: 'Graphite Anode Recipe Optimization',
  subtitle: 'Drakopoulos et al. (2021) • Formulation, Coating & Drying Parameters',
  targetDescription: 'Discharge Specific Capacity at Cycle 30 (D30)',
  targetUnit: 'mAh/g',
  evidenceKind: 'PHYSICAL_HISTORICAL',
  provenance: {
    authors: 'Drakopoulos et al.',
    title: 'Machine learning for the design and manufacturing of Li-ion battery electrodes',
    journal: 'Cell Reports Physical Science',
    year: 2021,
    doi: '10.1016/j.xcrp.2021.100683',
    license: 'CC BY 4.0',
    facility: 'Academic Battery Fabrication Facility',
    replicateSummary: '12 strictly complete manufacturing recipes, 36 individual full/half cells evaluated under cycling protocol'
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
        { name: 'active_material_pct', label: 'Active Material Fraction', unit: 'wt%', typicalRange: '93.0 - 96.0 wt%' },
        { name: 'carbon_black_pct', label: 'Super C65 Additive', unit: 'wt%', typicalRange: '1.5 - 3.5 wt%' },
        { name: 'binder_pct', label: 'CMC/SBR Binder System', unit: 'wt%', typicalRange: '2.5 - 3.5 wt%' }
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
        { name: 'mixing_speed_rpm', label: 'Impeller Rotation Speed', unit: 'RPM', typicalRange: '1200 - 2500 RPM' },
        { name: 'mixing_duration_min', label: 'Dispersion Duration', unit: 'min', typicalRange: '30 - 90 min' },
        { name: 'vacuum_pressure_mbar', label: 'Degassing Pressure', unit: 'mbar', typicalRange: '50 - 100 mbar' }
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
        { name: 'airflow_velocity_m_s', label: 'Convection Airflow', unit: 'm/s', typicalRange: '1.2 - 2.5 m/s' }
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
  candidates: [
    {
      id: 'protocol-c1c280b7366f',
      displayId: 'Recipe-01',
      controls: {
        coating_gap_um: 100.0,
        coating_speed_m_per_min: 0.2,
        drying_temperature_c: 60.0,
        calendering_applied: false
      },
      revealedTarget: {
        name: 'D30 Specific Capacity',
        value: 402.25,
        unit: 'mAh/g'
      },
      metrology: {
        activeMassMg: 7.70,
        calenderedThicknessUm: 37.5
      },
      replicateCount: 3,
      cellIds: ['ASC-52', 'ASC-53', 'ASC-54']
    },
    {
      id: 'protocol-fd150c39c22f',
      displayId: 'Recipe-02',
      controls: {
        coating_gap_um: 100.0,
        coating_speed_m_per_min: 0.4,
        drying_temperature_c: 60.0,
        calendering_applied: false
      },
      revealedTarget: {
        name: 'D30 Specific Capacity',
        value: 392.51,
        unit: 'mAh/g'
      },
      metrology: {
        activeMassMg: 8.15,
        calenderedThicknessUm: 38.0
      },
      replicateCount: 3,
      cellIds: ['ASC-58', 'ASC-59', 'ASC-60']
    },
    {
      id: 'protocol-0d38055e23a7',
      displayId: 'Recipe-03',
      controls: {
        coating_gap_um: 150.0,
        coating_speed_m_per_min: 0.1,
        drying_temperature_c: 60.0,
        calendering_applied: true
      },
      revealedTarget: {
        name: 'D30 Specific Capacity',
        value: 354.54,
        unit: 'mAh/g'
      },
      metrology: {
        activeMassMg: 11.10,
        calenderedThicknessUm: 42.0
      },
      replicateCount: 3,
      cellIds: ['ASC-40', 'ASC-41', 'ASC-42']
    },
    {
      id: 'protocol-7acc4561b3f7',
      displayId: 'Recipe-04',
      controls: {
        coating_gap_um: 150.0,
        coating_speed_m_per_min: 0.1,
        drying_temperature_c: 60.0,
        calendering_applied: false
      },
      revealedTarget: {
        name: 'D30 Specific Capacity',
        value: 316.39,
        unit: 'mAh/g'
      },
      metrology: {
        activeMassMg: 11.54,
        calenderedThicknessUm: 45.0
      },
      replicateCount: 3,
      cellIds: ['ASC-37', 'ASC-38', 'ASC-39']
    },
    {
      id: 'protocol-a4fcf0522485',
      displayId: 'Recipe-05',
      controls: {
        coating_gap_um: 150.0,
        coating_speed_m_per_min: 0.4,
        drying_temperature_c: 100.0,
        calendering_applied: false
      },
      revealedTarget: {
        name: 'D30 Specific Capacity',
        value: 265.59,
        unit: 'mAh/g'
      },
      metrology: {
        activeMassMg: 14.88,
        calenderedThicknessUm: 48.0
      },
      replicateCount: 3,
      cellIds: ['ASC-31', 'ASC-32', 'ASC-33']
    },
    {
      id: 'protocol-d3602183e567',
      displayId: 'Recipe-06',
      controls: {
        coating_gap_um: 100.0,
        coating_speed_m_per_min: 0.2,
        drying_temperature_c: 60.0,
        calendering_applied: true
      },
      revealedTarget: {
        name: 'D30 Specific Capacity',
        value: 263.89,
        unit: 'mAh/g'
      },
      metrology: {
        activeMassMg: 7.48,
        calenderedThicknessUm: 36.0
      },
      replicateCount: 3,
      cellIds: ['ASC-49', 'ASC-50', 'ASC-51']
    },
    {
      id: 'protocol-f3ae9bcb49de',
      displayId: 'Recipe-07',
      controls: {
        coating_gap_um: 150.0,
        coating_speed_m_per_min: 0.2,
        drying_temperature_c: 60.0,
        calendering_applied: false
      },
      revealedTarget: {
        name: 'D30 Specific Capacity',
        value: 241.91,
        unit: 'mAh/g'
      },
      metrology: {
        activeMassMg: 11.58,
        calenderedThicknessUm: 44.5
      },
      replicateCount: 3,
      cellIds: ['ASC-46', 'ASC-47', 'ASC-48']
    },
    {
      id: 'protocol-9ecd72d3534a',
      displayId: 'Recipe-08',
      controls: {
        coating_gap_um: 150.0,
        coating_speed_m_per_min: 0.4,
        drying_temperature_c: 60.0,
        calendering_applied: false
      },
      revealedTarget: {
        name: 'D30 Specific Capacity',
        value: 235.08,
        unit: 'mAh/g'
      },
      metrology: {
        activeMassMg: 15.50,
        calenderedThicknessUm: 49.0
      },
      replicateCount: 3,
      cellIds: ['ASC-16', 'ASC-17', 'ASC-18']
    },
    {
      id: 'protocol-1839e8c24871',
      displayId: 'Recipe-09',
      controls: {
        coating_gap_um: 150.0,
        coating_speed_m_per_min: 0.4,
        drying_temperature_c: 60.0,
        calendering_applied: true
      },
      revealedTarget: {
        name: 'D30 Specific Capacity',
        value: 140.35,
        unit: 'mAh/g'
      },
      metrology: {
        activeMassMg: 15.66,
        calenderedThicknessUm: 41.0
      },
      replicateCount: 3,
      cellIds: ['ASC-13', 'ASC-14', 'ASC-15']
    },
    {
      id: 'protocol-95a0b3eea94d',
      displayId: 'Recipe-10',
      controls: {
        coating_gap_um: 200.0,
        coating_speed_m_per_min: 0.4,
        drying_temperature_c: 100.0,
        calendering_applied: false
      },
      revealedTarget: {
        name: 'D30 Specific Capacity',
        value: 115.71,
        unit: 'mAh/g'
      },
      metrology: {
        activeMassMg: 18.52,
        calenderedThicknessUm: 65.0
      },
      replicateCount: 3,
      cellIds: ['ASC-82', 'ASC-83', 'ASC-84']
    },
    {
      id: 'protocol-23509c88f8bf',
      displayId: 'Recipe-11',
      controls: {
        coating_gap_um: 200.0,
        coating_speed_m_per_min: 0.4,
        drying_temperature_c: 80.0,
        calendering_applied: false
      },
      revealedTarget: {
        name: 'D30 Specific Capacity',
        value: 111.89,
        unit: 'mAh/g'
      },
      metrology: {
        activeMassMg: 19.15,
        calenderedThicknessUm: 67.0
      },
      replicateCount: 3,
      cellIds: ['ASC-79', 'ASC-80', 'ASC-81']
    },
    {
      id: 'protocol-c005aa958750',
      displayId: 'Recipe-12',
      controls: {
        coating_gap_um: 200.0,
        coating_speed_m_per_min: 0.4,
        drying_temperature_c: 90.0,
        calendering_applied: true
      },
      revealedTarget: {
        name: 'D30 Specific Capacity',
        value: 41.54,
        unit: 'mAh/g'
      },
      metrology: {
        activeMassMg: 18.07,
        calenderedThicknessUm: 52.0
      },
      replicateCount: 3,
      cellIds: ['ASC-88', 'ASC-89', 'ASC-90']
    }
  ],
  replayInitialIds: [
    'protocol-0d38055e23a7',
    'protocol-95a0b3eea94d',
    'protocol-23509c88f8bf'
  ],
  replaySteps: [
    {
      step: 1,
      selectedCandidateId: 'protocol-d3602183e567',
      selectedCandidateDisplay: 'Recipe-06',
      controlsSummary: {
        'Coating Gap': '100 µm',
        'Speed': '0.2 m/min',
        'Drying Temp': '60 °C',
        'Calendered': 'Yes'
      },
      predictedMean: 332.1,
      predictedStd: 68.4,
      acquisitionValue: 14.82,
      revealedTarget: 263.89,
      bestSoFar: 354.54,
      simpleRegret: 47.71,
      isOptimal: false,
      explanation: 'Surrogate explored smaller 100 µm gap configuration with calendering applied to test low active-mass hypothesis.'
    },
    {
      step: 2,
      selectedCandidateId: 'protocol-7acc4561b3f7',
      selectedCandidateDisplay: 'Recipe-04',
      controlsSummary: {
        'Coating Gap': '150 µm',
        'Speed': '0.1 m/min',
        'Drying Temp': '60 °C',
        'Calendered': 'No'
      },
      predictedMean: 348.6,
      predictedStd: 42.1,
      acquisitionValue: 12.05,
      revealedTarget: 316.39,
      bestSoFar: 354.54,
      simpleRegret: 47.71,
      isOptimal: false,
      explanation: 'Evaluated uncalendered 150 µm condition to test if mechanical compaction was inducing particle fractures.'
    },
    {
      step: 3,
      selectedCandidateId: 'protocol-fd150c39c22f',
      selectedCandidateDisplay: 'Recipe-02',
      controlsSummary: {
        'Coating Gap': '100 µm',
        'Speed': '0.4 m/min',
        'Drying Temp': '60 °C',
        'Calendered': 'No'
      },
      predictedMean: 371.4,
      predictedStd: 38.9,
      acquisitionValue: 21.40,
      revealedTarget: 392.51,
      bestSoFar: 392.51,
      simpleRegret: 9.74,
      isOptimal: false,
      explanation: 'Breakthrough candidate: 100 µm uncalendered layer delivered 392.5 mAh/g, significantly surpassing initial baseline (354.5 mAh/g).'
    },
    {
      step: 4,
      selectedCandidateId: 'protocol-c1c280b7366f',
      selectedCandidateDisplay: 'Recipe-01 (Optimal Discovery)',
      controlsSummary: {
        'Coating Gap': '100 µm',
        'Speed': '0.2 m/min',
        'Drying Temp': '60 °C',
        'Calendered': 'No'
      },
      predictedMean: 398.2,
      predictedStd: 22.7,
      acquisitionValue: 28.65,
      revealedTarget: 402.25,
      bestSoFar: 402.25,
      simpleRegret: 0.00,
      isOptimal: true,
      explanation: 'Optimal discovery! Reached maximum recorded specific capacity (402.25 mAh/g) by pairing 100 µm gap with gentle 0.2 m/min coating speed.'
    },
    {
      step: 5,
      selectedCandidateId: 'protocol-a4fcf0522485',
      selectedCandidateDisplay: 'Recipe-05',
      controlsSummary: {
        'Coating Gap': '150 µm',
        'Speed': '0.4 m/min',
        'Drying Temp': '100 °C',
        'Calendered': 'No'
      },
      predictedMean: 294.1,
      predictedStd: 31.5,
      acquisitionValue: 3.12,
      revealedTarget: 265.59,
      bestSoFar: 402.25,
      simpleRegret: 0.00,
      isOptimal: false,
      explanation: 'Final exploration step confirming performance bounds at higher drying temperatures (100 °C).'
    }
  ],
  benchmark: {
    policyId: 'AICOSCIENTIST_PROCESS_SURROGATE',
    policyName: 'AIcoScientist Process Surrogate Engine',
    hitAt1Pct: 30.0,
    hitAt3Pct: 50.0,
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
    scientificInsight: 'Graphite flakes align horizontally during doctor-blade shearing. Excessive calendering causes particle fracture, reducing lithium intercalation channels.'
  }
};

export const WARWICK_SCENARIO: ExhibitionScenario = {
  id: 'warwick_nmc622_calendering',
  title: 'NMC622 Pilot Calendering Optimization',
  subtitle: 'Warwick Manufacturing Group • Roll Temperature, Gap & Multi-Pass Regimes',
  targetDescription: '5C Fast Discharge vs 0.2C Nominal Rate Ratio (5C/0.2C)',
  targetUnit: 'dimensionless_ratio',
  evidenceKind: 'PILOT_LINE_HISTORICAL',
  provenance: {
    authors: 'University of Warwick (WMG Pilot Line)',
    title: 'Physical and Electrochemical Characteristics of Calendered NMC622 Electrodes at Pilot-Plant Scale',
    journal: 'University of Warwick Open Data Repository',
    year: 2020,
    doi: '10.17632/warwick.calendering.nmc622',
    license: 'CC BY 4.0',
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
        { name: 'nmc_ratio', label: 'NMC622 Active Ratio', unit: 'wt%', typicalRange: '90.0 wt%' },
        { name: 'pvdf_ratio', label: 'PVDF Binder', unit: 'wt%', typicalRange: '5.0 wt%' },
        { name: 'c65_ratio', label: 'C65 Conductive Carbon', unit: 'wt%', typicalRange: '5.0 wt%' }
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
      detailedDescription: 'Pilot-scale continuous coating across two distinct areal loading regimes: Low (122.5 gsm) and High (182.7 gsm).',
      isOptimizationTarget: false,
      controllableParameters: [
        { name: 'target_coating_weight_gsm', label: 'Target Loading Regime', unit: 'g/m²', typicalRange: '122.5 vs 182.7 gsm' }
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
        { name: 'roll_gap_um', label: 'Mechanical Roll Gap', unit: 'µm', typicalRange: '390 - 495 µm' },
        { name: 'number_of_passes', label: 'Pass Count', unit: 'passes', typicalRange: '1 - 3 passes' },
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
      detailedDescription: 'Precision 14.8 mm electrode disc harvesting and 2032 coin half-cell construction with lithium metal counter-electrode.',
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
  candidates: [
    {
      id: 'EXP_01',
      displayId: 'EXP_01 (Low Temp, Porous)',
      controls: {
        loading_regime: 'LOW',
        target_coating_weight_gsm: 122.48,
        roll_temperature_c: 85.0,
        target_density_g_cm3: 2.70,
        roll_gap_um: 475.0,
        number_of_passes: 2
      },
      revealedTarget: {
        name: '5C / 0.2C Rate Ratio',
        value: 0.7661,
        unit: 'dimensionless_ratio'
      },
      metrology: {
        preCalenderingThicknessUm: 52.66,
        calenderedThicknessUm: 46.17,
        calenderedDensityGCm3: 2.608,
        calenderedPorosityPct: 41.40,
        tensileStrengthKPa: 685.07
      },
      replicateCount: 3,
      cellIds: ['DD001', 'DD002', 'DD056']
    },
    {
      id: 'EXP_02',
      displayId: 'EXP_02 (Low Temp, Medium)',
      controls: {
        loading_regime: 'LOW',
        target_coating_weight_gsm: 122.48,
        roll_temperature_c: 85.0,
        target_density_g_cm3: 2.95,
        roll_gap_um: 462.0,
        number_of_passes: 1
      },
      revealedTarget: {
        name: '5C / 0.2C Rate Ratio',
        value: 0.7775,
        unit: 'dimensionless_ratio'
      },
      metrology: {
        preCalenderingThicknessUm: 52.50,
        calenderedThicknessUm: 41.67,
        calenderedDensityGCm3: 2.844,
        calenderedPorosityPct: 36.08,
        tensileStrengthKPa: 619.15
      },
      replicateCount: 3,
      cellIds: ['DD032', 'DD033', 'DD034']
    },
    {
      id: 'EXP_03',
      displayId: 'EXP_03',
      controls: {
        loading_regime: 'LOW',
        target_coating_weight_gsm: 122.48,
        roll_temperature_c: 85.0,
        target_density_g_cm3: 3.20,
        roll_gap_um: 458.0,
        number_of_passes: 3
      },
      revealedTarget: {
        name: '5C / 0.2C Rate Ratio',
        value: 0.7947,
        unit: 'dimensionless_ratio'
      },
      metrology: {
        preCalenderingThicknessUm: 52.50,
        calenderedThicknessUm: 39.17,
        calenderedDensityGCm3: 3.029,
        calenderedPorosityPct: 31.94,
        tensileStrengthKPa: 777.23
      },
      replicateCount: 3,
      cellIds: ['DD020', 'DD021', 'DD022']
    },
    {
      id: 'EXP_04',
      displayId: 'EXP_04',
      controls: {
        loading_regime: 'LOW',
        target_coating_weight_gsm: 122.48,
        roll_temperature_c: 120.0,
        target_density_g_cm3: 2.70,
        roll_gap_um: 400.0,
        number_of_passes: 1
      },
      revealedTarget: {
        name: '5C / 0.2C Rate Ratio',
        value: 0.7389,
        unit: 'dimensionless_ratio'
      },
      metrology: {
        preCalenderingThicknessUm: 52.83,
        calenderedThicknessUm: 45.83,
        calenderedDensityGCm3: 2.684,
        calenderedPorosityPct: 39.70,
        tensileStrengthKPa: 683.73
      },
      replicateCount: 3,
      cellIds: ['DD013', 'DD015', 'DD057']
    },
    {
      id: 'EXP_05',
      displayId: 'EXP_05',
      controls: {
        loading_regime: 'LOW',
        target_coating_weight_gsm: 122.48,
        roll_temperature_c: 120.0,
        target_density_g_cm3: 2.95,
        roll_gap_um: 395.0,
        number_of_passes: 1
      },
      revealedTarget: {
        name: '5C / 0.2C Rate Ratio',
        value: 0.7677,
        unit: 'dimensionless_ratio'
      },
      metrology: {
        preCalenderingThicknessUm: 52.80,
        calenderedThicknessUm: 42.00,
        calenderedDensityGCm3: 2.882,
        calenderedPorosityPct: 35.24,
        tensileStrengthKPa: 741.60
      },
      replicateCount: 3,
      cellIds: ['DD035', 'DD036', 'DD037']
    },
    {
      id: 'EXP_06',
      displayId: 'EXP_06 (High Temp, Dense)',
      controls: {
        loading_regime: 'LOW',
        target_coating_weight_gsm: 122.48,
        roll_temperature_c: 120.0,
        target_density_g_cm3: 3.20,
        roll_gap_um: 390.0,
        number_of_passes: 2
      },
      revealedTarget: {
        name: '5C / 0.2C Rate Ratio',
        value: 0.7884,
        unit: 'dimensionless_ratio'
      },
      metrology: {
        preCalenderingThicknessUm: 51.83,
        calenderedThicknessUm: 38.50,
        calenderedDensityGCm3: 3.086,
        calenderedPorosityPct: 30.64,
        tensileStrengthKPa: 758.56
      },
      replicateCount: 3,
      cellIds: ['DD038', 'DD039', 'DD040']
    },
    {
      id: 'EXP_07',
      displayId: 'EXP_07',
      controls: {
        loading_regime: 'LOW',
        target_coating_weight_gsm: 122.48,
        roll_temperature_c: 145.0,
        target_density_g_cm3: 2.70,
        roll_gap_um: 460.0,
        number_of_passes: 1
      },
      revealedTarget: {
        name: '5C / 0.2C Rate Ratio',
        value: 0.6876,
        unit: 'dimensionless_ratio'
      },
      metrology: {
        preCalenderingThicknessUm: 53.20,
        calenderedThicknessUm: 45.83,
        calenderedDensityGCm3: 2.698,
        calenderedPorosityPct: 39.38,
        tensileStrengthKPa: 653.60
      },
      replicateCount: 3,
      cellIds: ['DD004', 'DD005', 'DD006']
    },
    {
      id: 'EXP_08',
      displayId: 'EXP_08',
      controls: {
        loading_regime: 'LOW',
        target_coating_weight_gsm: 122.48,
        roll_temperature_c: 145.0,
        target_density_g_cm3: 2.95,
        roll_gap_um: 450.0,
        number_of_passes: 1
      },
      revealedTarget: {
        name: '5C / 0.2C Rate Ratio',
        value: 0.7777,
        unit: 'dimensionless_ratio'
      },
      metrology: {
        preCalenderingThicknessUm: 52.83,
        calenderedThicknessUm: 42.00,
        calenderedDensityGCm3: 2.884,
        calenderedPorosityPct: 35.18,
        tensileStrengthKPa: 780.96
      },
      replicateCount: 3,
      cellIds: ['DD041', 'DD042', 'DD043']
    },
    {
      id: 'EXP_09',
      displayId: 'EXP_09',
      controls: {
        loading_regime: 'LOW',
        target_coating_weight_gsm: 122.48,
        roll_temperature_c: 145.0,
        target_density_g_cm3: 3.20,
        roll_gap_um: 446.0,
        number_of_passes: 1
      },
      revealedTarget: {
        name: '5C / 0.2C Rate Ratio',
        value: 0.7737,
        unit: 'dimensionless_ratio'
      },
      metrology: {
        preCalenderingThicknessUm: 53.00,
        calenderedThicknessUm: 39.67,
        calenderedDensityGCm3: 3.088,
        calenderedPorosityPct: 30.61,
        tensileStrengthKPa: 745.65
      },
      replicateCount: 3,
      cellIds: ['DD023', 'DD024', 'DD025']
    },
    {
      id: 'EXP_10',
      displayId: 'EXP_10 (High Loading)',
      controls: {
        loading_regime: 'HIGH',
        target_coating_weight_gsm: 182.73,
        roll_temperature_c: 85.0,
        target_density_g_cm3: 2.70,
        roll_gap_um: 495.0,
        number_of_passes: 1
      },
      revealedTarget: {
        name: '5C / 0.2C Rate Ratio',
        value: 0.0988,
        unit: 'dimensionless_ratio'
      },
      metrology: {
        preCalenderingThicknessUm: 75.33,
        calenderedThicknessUm: 67.83,
        calenderedDensityGCm3: 2.656,
        calenderedPorosityPct: 40.32,
        tensileStrengthKPa: 660.05
      },
      replicateCount: 3,
      cellIds: ['DD059', 'DD027', 'DD028']
    },
    {
      id: 'EXP_11',
      displayId: 'EXP_11',
      controls: {
        loading_regime: 'HIGH',
        target_coating_weight_gsm: 182.73,
        roll_temperature_c: 85.0,
        target_density_g_cm3: 2.95,
        roll_gap_um: 476.0,
        number_of_passes: 1
      },
      revealedTarget: {
        name: '5C / 0.2C Rate Ratio',
        value: 0.4860,
        unit: 'dimensionless_ratio'
      },
      metrology: {
        preCalenderingThicknessUm: 74.66,
        calenderedThicknessUm: 61.50,
        calenderedDensityGCm3: 2.915,
        calenderedPorosityPct: 34.49,
        tensileStrengthKPa: 592.64
      },
      replicateCount: 3,
      cellIds: ['DD053', 'DD055', 'DD058']
    },
    {
      id: 'EXP_12',
      displayId: 'EXP_12',
      controls: {
        loading_regime: 'HIGH',
        target_coating_weight_gsm: 182.73,
        roll_temperature_c: 85.0,
        target_density_g_cm3: 3.20,
        roll_gap_um: 466.0,
        number_of_passes: 2
      },
      revealedTarget: {
        name: '5C / 0.2C Rate Ratio',
        value: 0.3031,
        unit: 'dimensionless_ratio'
      },
      metrology: {
        preCalenderingThicknessUm: 75.00,
        calenderedThicknessUm: 57.00,
        calenderedDensityGCm3: 3.109,
        calenderedPorosityPct: 30.13,
        tensileStrengthKPa: 609.17
      },
      replicateCount: 3,
      cellIds: ['DD007', 'DD008', 'DD009']
    },
    {
      id: 'EXP_13',
      displayId: 'EXP_13 (Initial)',
      controls: {
        loading_regime: 'HIGH',
        target_coating_weight_gsm: 182.73,
        roll_temperature_c: 120.0,
        target_density_g_cm3: 2.70,
        roll_gap_um: 420.0,
        number_of_passes: 2
      },
      revealedTarget: {
        name: '5C / 0.2C Rate Ratio',
        value: 0.2818,
        unit: 'dimensionless_ratio'
      },
      metrology: {
        preCalenderingThicknessUm: 73.00,
        calenderedThicknessUm: 68.33,
        calenderedDensityGCm3: 2.707,
        calenderedPorosityPct: 39.17,
        tensileStrengthKPa: 698.03
      },
      replicateCount: 3,
      cellIds: ['DD050', 'DD051', 'DD052']
    },
    {
      id: 'EXP_14',
      displayId: 'EXP_14 (Initial)',
      controls: {
        loading_regime: 'HIGH',
        target_coating_weight_gsm: 182.73,
        roll_temperature_c: 120.0,
        target_density_g_cm3: 2.95,
        roll_gap_um: 410.0,
        number_of_passes: 2
      },
      revealedTarget: {
        name: '5C / 0.2C Rate Ratio',
        value: 0.4214,
        unit: 'dimensionless_ratio'
      },
      metrology: {
        preCalenderingThicknessUm: 75.50,
        calenderedThicknessUm: 62.67,
        calenderedDensityGCm3: 2.901,
        calenderedPorosityPct: 34.80,
        tensileStrengthKPa: 632.69
      },
      replicateCount: 3,
      cellIds: ['DD047', 'DD048', 'DD049']
    },
    {
      id: 'EXP_15',
      displayId: 'EXP_15',
      controls: {
        loading_regime: 'HIGH',
        target_coating_weight_gsm: 182.73,
        roll_temperature_c: 120.0,
        target_density_g_cm3: 3.20,
        roll_gap_um: 400.0,
        number_of_passes: 2
      },
      revealedTarget: {
        name: '5C / 0.2C Rate Ratio',
        value: 0.3682,
        unit: 'dimensionless_ratio'
      },
      metrology: {
        preCalenderingThicknessUm: 75.33,
        calenderedThicknessUm: 57.83,
        calenderedDensityGCm3: 3.105,
        calenderedPorosityPct: 30.22,
        tensileStrengthKPa: 634.35
      },
      replicateCount: 3,
      cellIds: ['DD016', 'DD017', 'DD018']
    },
    {
      id: 'EXP_16',
      displayId: 'EXP_16',
      controls: {
        loading_regime: 'HIGH',
        target_coating_weight_gsm: 182.73,
        roll_temperature_c: 145.0,
        target_density_g_cm3: 2.70,
        roll_gap_um: 480.0,
        number_of_passes: 1
      },
      revealedTarget: {
        name: '5C / 0.2C Rate Ratio',
        value: 0.1738,
        unit: 'dimensionless_ratio'
      },
      metrology: {
        preCalenderingThicknessUm: 72.40,
        calenderedThicknessUm: 67.67,
        calenderedDensityGCm3: 2.681,
        calenderedPorosityPct: 39.75,
        tensileStrengthKPa: 652.16
      },
      replicateCount: 3,
      cellIds: ['DD029', 'DD030', 'DD031']
    },
    {
      id: 'EXP_17',
      displayId: 'EXP_17',
      controls: {
        loading_regime: 'HIGH',
        target_coating_weight_gsm: 182.73,
        roll_temperature_c: 145.0,
        target_density_g_cm3: 2.95,
        roll_gap_um: 473.0,
        number_of_passes: 3
      },
      revealedTarget: {
        name: '5C / 0.2C Rate Ratio',
        value: 0.4201,
        unit: 'dimensionless_ratio'
      },
      metrology: {
        preCalenderingThicknessUm: 75.00,
        calenderedThicknessUm: 63.17,
        calenderedDensityGCm3: 2.892,
        calenderedPorosityPct: 35.01,
        tensileStrengthKPa: 687.36
      },
      replicateCount: 3,
      cellIds: ['DD044', 'DD045', 'DD046']
    },
    {
      id: 'EXP_18',
      displayId: 'EXP_18 (Initial)',
      controls: {
        loading_regime: 'HIGH',
        target_coating_weight_gsm: 182.73,
        roll_temperature_c: 145.0,
        target_density_g_cm3: 3.20,
        roll_gap_um: 460.0,
        number_of_passes: 2
      },
      revealedTarget: {
        name: '5C / 0.2C Rate Ratio',
        value: 0.2816,
        unit: 'dimensionless_ratio'
      },
      metrology: {
        preCalenderingThicknessUm: 74.00,
        calenderedThicknessUm: 57.33,
        calenderedDensityGCm3: 3.139,
        calenderedPorosityPct: 29.47,
        tensileStrengthKPa: 705.60
      },
      replicateCount: 3,
      cellIds: ['DD010', 'DD011', 'DD012']
    }
  ],
  replayInitialIds: ['EXP_14', 'EXP_18', 'EXP_13'],
  replaySteps: [
    {
      step: 1,
      selectedCandidateId: 'EXP_15',
      selectedCandidateDisplay: 'EXP_15',
      controlsSummary: {
        'Roll Temp': '120 °C',
        'Roll Gap': '400 µm',
        'Passes': '2 passes',
        'Loading': 'High',
        'Target Density': '3.20 g/cm³'
      },
      predictedMean: 0.3283,
      predictedStd: 0.0949,
      acquisitionValue: 0.0082,
      revealedTarget: 0.3682,
      bestSoFar: 0.4214,
      simpleRegret: 0.3733,
      isOptimal: false,
      explanation: 'Surrogate explored high-density regime at 120 °C. The revealed 5C rate retention (0.368) proved that high thickness impedes ion diffusion.',
      intermediateMetrology: {
        density: '3.105 g/cm³',
        porosity: '30.22%',
        thickness: '57.8 µm'
      }
    },
    {
      step: 2,
      selectedCandidateId: 'EXP_01',
      selectedCandidateDisplay: 'EXP_01',
      controlsSummary: {
        'Roll Temp': '85 °C',
        'Roll Gap': '475 µm',
        'Passes': '2 passes',
        'Loading': 'Low',
        'Target Density': '2.70 g/cm³'
      },
      predictedMean: 0.5891,
      predictedStd: 0.1142,
      acquisitionValue: 0.0812,
      revealedTarget: 0.7661,
      bestSoFar: 0.7661,
      simpleRegret: 0.0287,
      isOptimal: false,
      explanation: 'Pivoted to lower loading regime (122.5 gsm) and lower roll temperature (85 °C). Jump in rate retention to 0.7661 (+81% improvement).',
      intermediateMetrology: {
        density: '2.608 g/cm³',
        porosity: '41.40%',
        thickness: '46.2 µm'
      }
    },
    {
      step: 3,
      selectedCandidateId: 'EXP_02',
      selectedCandidateDisplay: 'EXP_02',
      controlsSummary: {
        'Roll Temp': '85 °C',
        'Roll Gap': '462 µm',
        'Passes': '1 pass',
        'Loading': 'Low',
        'Target Density': '2.95 g/cm³'
      },
      predictedMean: 0.7245,
      predictedStd: 0.0683,
      acquisitionValue: 0.0435,
      revealedTarget: 0.7775,
      bestSoFar: 0.7775,
      simpleRegret: 0.0172,
      isOptimal: false,
      explanation: 'Exploiting 85 °C sweet spot at intermediate density (2.84 g/cm³), establishing strong electronic percolation without pore blockage.',
      intermediateMetrology: {
        density: '2.844 g/cm³',
        porosity: '36.08%',
        thickness: '41.7 µm'
      }
    },
    {
      step: 4,
      selectedCandidateId: 'EXP_03',
      selectedCandidateDisplay: 'EXP_03 (Optimal Discovery)',
      controlsSummary: {
        'Roll Temp': '85 °C',
        'Roll Gap': '458 µm',
        'Passes': '3 passes',
        'Loading': 'Low',
        'Target Density': '3.20 g/cm³'
      },
      predictedMean: 0.7812,
      predictedStd: 0.0391,
      acquisitionValue: 0.0384,
      revealedTarget: 0.7947,
      bestSoFar: 0.7947,
      simpleRegret: 0.0000,
      isOptimal: true,
      explanation: 'Optimal pilot condition discovered! 85 °C roll temperature with 3 gentle compaction passes achieves ideal balance of high density (3.029 g/cm³) and ionic permeability (31.94% porosity).',
      intermediateMetrology: {
        density: '3.029 g/cm³',
        porosity: '31.94%',
        thickness: '39.2 µm'
      }
    },
    {
      step: 5,
      selectedCandidateId: 'EXP_06',
      selectedCandidateDisplay: 'EXP_06',
      controlsSummary: {
        'Roll Temp': '120 °C',
        'Roll Gap': '390 µm',
        'Passes': '2 passes',
        'Loading': 'Low',
        'Target Density': '3.20 g/cm³'
      },
      predictedMean: 0.7640,
      predictedStd: 0.0415,
      acquisitionValue: 0.0121,
      revealedTarget: 0.7884,
      bestSoFar: 0.7947,
      simpleRegret: 0.0000,
      isOptimal: false,
      explanation: 'Confirmatory boundary check at 120 °C. Revealed 0.7884, confirming 85 °C remains superior.',
      intermediateMetrology: {
        density: '3.086 g/cm³',
        porosity: '30.64%',
        thickness: '38.5 µm'
      }
    }
  ],
  benchmark: {
    policyId: 'AICOSCIENTIST_FULL_PROCESS_ENGINE',
    policyName: 'AIcoScientist Process Optimization Engine',
    hitAt1Pct: 0.0,
    hitAt3Pct: 50.0,
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
    scientificInsight: 'Calendering compacts spherical NMC particles into close electronic contact. If over-compressed below 25% porosity, liquid electrolyte cannot penetrate tortuous pore paths, severely degrading fast-charging rate capability.'
  }
};
