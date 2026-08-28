param(
    [string]$OutputFile = "F:\AI\GTIP\outputs\Mau_nhap_du_lieu_vat_lieu_cho_AI_v2.xlsx"
)

$ErrorActionPreference = "Stop"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom

function Invoke-OfficeBatch {
    param([System.Collections.IList]$Operations)
    for ($i = 0; $i -lt $Operations.Count; $i += 45) {
        $end = [Math]::Min($i + 44, $Operations.Count - 1)
        $chunk = @($Operations[$i..$end])
        $json = $chunk | ConvertTo-Json -Depth 12 -Compress
        $result = $json | officecli batch $OutputFile --json 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "officecli batch failed: $($result -join [Environment]::NewLine)"
        }
    }
}

function Op-Set([string]$Path, [hashtable]$Props) {
    return @{ command = "set"; path = $Path; props = $Props }
}

function Op-Add([string]$Parent, [string]$Type, [hashtable]$Props) {
    return @{ command = "add"; parent = $Parent; type = $Type; props = $Props }
}

function Get-ExcelColumn([int]$Number) {
    $name = ""
    while ($Number -gt 0) {
        $Number--
        $name = [char](65 + ($Number % 26)) + $name
        $Number = [Math]::Floor($Number / 26)
    }
    return $name
}

function Get-Unit([string]$Field) {
    if ($Field -match "_wt_pct$|_pct$|retention|efficiency|coverage|improvement") { return "%" }
    if ($Field -match "_mah_g$") { return "mAh g^-1" }
    if ($Field -match "_wh_kg$") { return "Wh kg^-1" }
    if ($Field -match "_mg_cm2$") { return "mg cm^-2" }
    if ($Field -match "_cm2$") { return "cm^2" }
    if ($Field -match "_nm$") { return "nm" }
    if ($Field -match "_um$") { return "um" }
    if ($Field -match "_mpa$") { return "MPa" }
    if ($Field -match "_mpas$") { return "mPa.s" }
    if ($Field -match "_rpm$") { return "rpm" }
    if ($Field -match "_mm_s$") { return "mm s^-1" }
    if ($Field -match "_mm$") { return "mm" }
    if ($Field -match "_min$") { return "min" }
    if ($Field -match "_h$") { return "h" }
    if ($Field -match "_c$") { return "degC" }
    if ($Field -match "_v$") { return "V" }
    if ($Field -match "_mv$") { return "mV" }
    if ($Field -match "_hz$") { return "Hz" }
    if ($Field -match "_ul$") { return "uL" }
    if ($Field -match "_x$") { return "x" }
    if ($Field -match "date|timestamp|_at$") { return "ISO date/time" }
    if ($Field -match "^is_|eligible|censored|usable") { return "Yes/No" }
    return "Text / number"
}

function Get-Role([string]$Sheet, [string]$Field) {
    if ($Field -match "_id$|^.*_id_") { return "ID / foreign key" }
    if ($Field -match "predicted|acquisition|novelty|interval") { return "Model output" }
    if ($Field -match "capacity|retention|efficiency|rct|specific_energy|average_voltage") { return "Outcome / measurement" }
    if ($Field -match "qc|status|failure|decision|eligible|censored|usable|pass_fail") { return "QC / governance" }
    if ($Field -match "path|file|sha256|version|date|timestamp|operator|analyst|reviewer|created|lot") { return "Traceability" }
    if ($Sheet -match "SEM|EDX|Cycle") { return "Measurement" }
    return "Controllable input / metadata"
}

function Get-Priority([string]$Field) {
    if ($Field -match "notes|scientific_reason|corrective_action|working_distance|average_voltage|specific_energy") { return "P2" }
    if ($Field -match "SEM|EDX|particle|porosity|agglomeration|crack|uniformity|early_prediction|viscosity|oxide|flake|layer|termination|rct|adhesion|conductivity") { return "P1" }
    return "P0"
}

function Get-Rule([string]$Field) {
    if ($Field -match "_id$") { return "Không để trống khi dùng; ID duy nhất hoặc khóa ngoại hợp lệ" }
    if ($Field -match "content_wt_pct") { return "0-100; tổng bốn thành phần = 100 +/- 0.1" }
    if ($Field -match "score|index|density$|uniformity") { return "0-1 nếu là chỉ số chuẩn hóa" }
    if ($Field -match "_pct$|retention|efficiency|coverage") { return "0-100" }
    if ($Field -match "path|file") { return "Đường dẫn phải tồn tại; không đổi tên file sau khi ghi" }
    if ($Field -match "date|timestamp|_at$") { return "Dùng ngày/giờ ISO; không nhập chuỗi mơ hồ" }
    if ($Field -match "status|qc|decision|category|basis|stage|source") { return "Chọn từ danh mục chuẩn" }
    return "Một cột một kiểu dữ liệu; không trộn đơn vị"
}

$schemas = [ordered]@{
    "02_Recipe" = @(
        "recipe_id","recipe_version","recipe_status","recipe_source","si_lot_id","mxene_lot_id","alginate_lot_id","carbon_lot_id",
        "si_content_wt_pct","mxene_content_wt_pct","alginate_content_wt_pct","carbon_content_wt_pct","composition_sum_wt_pct","composition_qc",
        "formulation_notes","created_at","created_by","optimization_eligible"
    )
    "03_Material_Lots" = @(
        "material_lot_id","material_role","material_name","supplier","product_code","lot_number","received_date","material_status",
        "si_d50_nm","si_d90_nm","si_surface_oxide_pct","mxene_flake_size_um","mxene_layer_count","mxene_termination_category","mxene_oxidation_pct",
        "carbon_type","binder_grade","storage_condition","characterization_file_path","certificate_file_path","notes"
    )
    "04_Electrode_Process" = @(
        "electrode_id","recipe_id","batch_id","replicate_id","operator","fabrication_date","process_status","solvent","solid_loading_pct","solvent_ratio",
        "mixing_sequence","mixer_type","mixing_speed_rpm","mixing_time_min","slurry_temperature_c","rest_time_min","degassing_method","slurry_viscosity_mpas",
        "coating_method","coating_speed_mm_s","coating_gap_um","drying_temp_c","drying_time_h","vacuum_condition","pressing_pressure_mpa",
        "electrode_thickness_um","mass_loading_mg_cm2","electrode_porosity_pct","active_area_cm2","current_collector","process_notes"
    )
    "05_Cell_Assembly" = @(
        "cell_id","electrode_id","cell_batch_id","replicate_id","assembly_date","operator","assembly_status","cell_type","counter_electrode","reference_electrode",
        "separator","electrolyte_name","electrolyte_lot_id","electrolyte_volume_ul","n_p_ratio","rest_time_h","crimp_pressure","crimp_pressure_unit",
        "assembly_environment","notes"
    )
    "06_Test_Protocol" = @(
        "protocol_id","cell_id","protocol_version","protocol_status","test_start_date","cycler_id","channel_id","formation_protocol","formation_cycles",
        "charge_current_density_ma_g","discharge_current_density_ma_g","c_rate","voltage_min_v","voltage_max_v","test_temperature_c","rest_between_steps_min",
        "eis_frequency_min_hz","eis_frequency_max_hz","eis_amplitude_mv","stop_rule","notes"
    )
    "07_Cycle_Data" = @(
        "measurement_id","cell_id","protocol_id","cycle_index","step_type","charge_capacity_mah_g","discharge_capacity_mah_g","coulombic_efficiency_pct",
        "specific_energy_wh_kg","average_voltage_v","charge_time_min","discharge_time_min","test_timestamp","raw_file_path","qc_flag","notes"
    )
    "08_Electrochem_Summary" = @(
        "cell_id","recipe_id","electrode_id","protocol_id","initial_discharge_capacity_mah_g","initial_coulombic_efficiency_pct","capacity_10_mah_g","capacity_20_mah_g",
        "capacity_50_mah_g","capacity_100_mah_g","retention_50_pct","retention_100_pct","avg_ce_cycle_2_100_pct","rct_initial_ohm","rct_after_100_ohm",
        "capacity_fade_mah_g","target_status","early_prediction_mean_pct","early_prediction_std_pct","summary_qc","raw_file_path","calculated_at"
    )
    "09_SEM_Measurements" = @(
        "sem_measurement_id","electrode_id","cell_id","measurement_stage","roi_id","image_file_path","image_sha256","magnification_x","accelerating_voltage_kv",
        "working_distance_mm","pixel_size","pixel_size_unit","particle_size_mean_nm","particle_size_std_nm","porosity_score","agglomeration_index","crack_density",
        "surface_uniformity","analysis_method","analysis_version","analyst","measurement_date","qc_flag","notes"
    )
    "10_EDX_Measurements" = @(
        "edx_measurement_id","electrode_id","cell_id","measurement_stage","roi_id","raw_file_path","edx_basis","si_pct","ti_pct","c_pct","o_pct","impurity_pct",
        "si_ti_ratio","c_o_ratio","elemental_uniformity_score","accelerating_voltage_kv","analysis_method","analysis_version","analyst","measurement_date","qc_flag","notes"
    )
    "11_Recommendation_Log" = @(
        "recommendation_id","model_version","model_timestamp","training_data_cutoff","acquisition_function","rank","candidate_recipe_id","si_content_wt_pct",
        "mxene_content_wt_pct","alginate_content_wt_pct","carbon_content_wt_pct","mixing_time_min","drying_temp_c","pressing_pressure_mpa",
        "predicted_retention_mean_pct","predicted_retention_std_pct","lower_95_pct","upper_95_pct","acquisition_score","novelty_distance","constraint_status",
        "recommendation_type","scientific_reason","reviewer_decision","reviewer","decision_date","executed_cell_ids","observed_retention_mean_pct",
        "observed_retention_std_pct","outcome_status","notes"
    )
    "12_Validation_Log" = @(
        "validation_id","model_version","validation_type","split_group","n_train","n_test","mae","rmse","r2","spearman_rank","top_k_hit_rate_pct",
        "interval_80_coverage_pct","interval_95_coverage_pct","calibration_error","baseline_method","baseline_score","ai_score","improvement_pct","locked_at",
        "evaluated_at","evaluator","evidence_file_path","pass_fail","notes"
    )
    "13_Failure_Log" = @(
        "failure_id","entity_type","entity_id","recipe_id","failure_stage","failure_category","failure_reason","censored","usable_for_model",
        "detected_at","detected_by","corrective_action","raw_file_path","notes"
    )
}

$maxRows = @{
    "02_Recipe" = 201; "03_Material_Lots" = 201; "04_Electrode_Process" = 201; "05_Cell_Assembly" = 201; "06_Test_Protocol" = 201;
    "07_Cycle_Data" = 1001; "08_Electrochem_Summary" = 201; "09_SEM_Measurements" = 501; "10_EDX_Measurements" = 501;
    "11_Recommendation_Log" = 201; "12_Validation_Log" = 201; "13_Failure_Log" = 201
}

$allSheets = @("00_Huong_dan","01_Danh_muc_truong") + @($schemas.Keys) + @("14_QC_Dashboard","15_Lookups")

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputFile) | Out-Null
officecli create $OutputFile | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Không thể tạo workbook" }
officecli open $OutputFile | Out-Null

$ops = [System.Collections.ArrayList]::new()
[void]$ops.Add((Op-Set "/Sheet1" @{ name = "00_Huong_dan" }))
foreach ($sheet in $allSheets[1..($allSheets.Count - 1)]) {
    [void]$ops.Add((Op-Add "/" "sheet" @{ name = $sheet; tabColor = "1F4E78" }))
}
Invoke-OfficeBatch $ops

# Lookup lists
$lookups = [ordered]@{
    "RecipeStatus" = @("planned","approved","fabricated","completed","rejected","failed")
    "YesNo" = @("Yes","No")
    "RecipeSource" = @("initial_doe","AI","expert","control","literature")
    "MaterialRole" = @("Si","MXene","Alginate","Carbon","Electrolyte","Separator","Other")
    "CellType" = @("coin","pouch","swagelok","three_electrode","other")
    "MeasurementStage" = @("pristine","post_formation","post_cycle_10","post_cycle_50","post_cycle_100","post_mortem")
    "QCFlag" = @("PASS","CHECK","FAIL")
    "TargetStatus" = @("observed","early_predicted","censored","not_available")
    "RecommendationType" = @("exploit","explore","control","boundary")
    "ReviewerDecision" = @("accepted","modified","rejected","pending")
    "FailureCategory" = @("fabrication","assembly","cycling","measurement","data","other")
    "EDXBasis" = @("at_pct","wt_pct")
    "GenericStatus" = @("planned","active","completed","failed","cancelled")
    "StepType" = @("charge","discharge","rest","eis","other")
    "ValidationType" = @("group_cv","leave_one_batch_out","prospective","external")
}
$ops = [System.Collections.ArrayList]::new()
$lookupCol = 1
foreach ($entry in $lookups.GetEnumerator()) {
    $col = Get-ExcelColumn $lookupCol
    [void]$ops.Add((Op-Set "/15_Lookups/$($col)1" @{ value = $entry.Key; type = "string" }))
    for ($i = 0; $i -lt $entry.Value.Count; $i++) {
        [void]$ops.Add((Op-Set "/15_Lookups/$col$($i + 2)" @{ value = $entry.Value[$i]; type = "string" }))
    }
    $lookupCol++
}
Invoke-OfficeBatch $ops

# Guide sheet
$guide = @(
    @{ Cell="A1"; Value="MẪU NHẬP DỮ LIỆU VẬT LIỆU V2 → CLOSED-LOOP AI CO-SCIENTIST" },
    @{ Cell="A3"; Value="Mục tiêu" },
    @{ Cell="B3"; Value="Thu thập dữ liệu có truy vết để AI đề xuất recipe tiếp theo, định lượng uncertainty và kiểm chứng prospective trước tháng 02/2027." },
    @{ Cell="A5"; Value="Kiến trúc ID" },
    @{ Cell="B5"; Value="recipe_id → electrode_id → cell_id → measurement_id; replicate và batch phải được lưu riêng, không gộp vào sample_id." },
    @{ Cell="A7"; Value="Nguyên tắc 1" },
    @{ Cell="B7"; Value="Chỉ dùng biến biết trước thí nghiệm để recommend. SEM/EDX và điện hóa là dữ liệu đo sau, dùng cho explanation hoặc model hai tầng." },
    @{ Cell="A8"; Value="Nguyên tắc 2" },
    @{ Cell="B8"; Value="Raw data không bị sửa. Chỉ số retention và ratio được tính trong summary/processed layer." },
    @{ Cell="A9"; Value="Nguyên tắc 3" },
    @{ Cell="B9"; Value="Mỗi vòng AI phải ghi recommendation trước khi có kết quả và lưu model_version, training cutoff, reviewer decision." },
    @{ Cell="A10"; Value="Nguyên tắc 4" },
    @{ Cell="B10"; Value="Không xóa thí nghiệm thất bại. Ghi vào Failure_Log để tránh selection bias và hỗ trợ constrained optimization." },
    @{ Cell="A12"; Value="Quy trình nhập" },
    @{ Cell="B12"; Value="1) Material lots → 2) Recipe → 3) Electrode process → 4) Cell assembly → 5) Test protocol → 6) Cycle/SEM/EDX → 7) Summary → 8) Recommendation/validation." },
    @{ Cell="A14"; Value="Mức ưu tiên" },
    @{ Cell="B14"; Value="P0: bắt buộc cho closed-loop; P1: tăng chất lượng model/cơ chế; P2: bổ sung khi có điều kiện. Xem 01_Danh_muc_truong." },
    @{ Cell="A16"; Value="Bảng" },
    @{ Cell="B16"; Value="Vai trò" }
)
$sheetDescriptions = [ordered]@{
    "02_Recipe"="Công thức và lô vật liệu; composition closure được tính tự động."
    "03_Material_Lots"="Đặc trưng từng lô Si/MXene/binder/carbon để chống batch confounding."
    "04_Electrode_Process"="Slurry, coating, drying, pressing và đặc trưng electrode."
    "05_Cell_Assembly"="Cell/replicate, electrolyte, separator và điều kiện assembly."
    "06_Test_Protocol"="Formation, C-rate/current density, voltage window và nhiệt độ test."
    "07_Cycle_Data"="Dữ liệu dài theo từng cycle; đây là raw source cho early prediction."
    "08_Electrochem_Summary"="Target và metric tổng hợp; retention/capacity fade tính tự động."
    "09_SEM_Measurements"="Một hàng cho mỗi ROI/stage, có metadata ảnh và version thuật toán."
    "10_EDX_Measurements"="Một hàng cho mỗi ROI/stage, giữ basis at.% hoặc wt.% rõ ràng."
    "11_Recommendation_Log"="Bất biến quyết định AI, uncertainty, reviewer và outcome thực tế."
    "12_Validation_Log"="Group CV, prospective validation, calibration và so sánh baseline."
    "13_Failure_Log"="Thất bại/censored data, nguyên nhân và khả năng dùng cho model."
    "14_QC_Dashboard"="Kiểm tra số lượng, coverage và cảnh báo trước khi train."
}
$ops = [System.Collections.ArrayList]::new()
foreach ($item in $guide) { [void]$ops.Add((Op-Set "/00_Huong_dan/$($item.Cell)" @{ value=$item.Value; type="string" })) }
$row = 17
foreach ($entry in $sheetDescriptions.GetEnumerator()) {
    [void]$ops.Add((Op-Set "/00_Huong_dan/A$row" @{ value=$entry.Key; type="string" }))
    [void]$ops.Add((Op-Set "/00_Huong_dan/B$row" @{ value=$entry.Value; type="string" }))
    $row++
}
Invoke-OfficeBatch $ops

# Data sheets headers and base formatting
foreach ($sheet in $schemas.Keys) {
    $headers = $schemas[$sheet]
    $lastCol = Get-ExcelColumn $headers.Count
    $limit = $maxRows[$sheet]
    $ops = [System.Collections.ArrayList]::new()
    for ($i = 0; $i -lt $headers.Count; $i++) {
        $col = Get-ExcelColumn ($i + 1)
        [void]$ops.Add((Op-Set "/$sheet/$($col)1" @{ value=$headers[$i]; type="string" }))
    }
    [void]$ops.Add((Op-Set "/$sheet/A1:$($lastCol)1" @{ fill="1F4E78"; "font.color"="FFFFFF"; "font.bold"=$true; "font.name"="Aptos"; "alignment.wrapText"=$true; "alignment.vertical"="center"; border="thin"; "border.color"="D9E2F3" }))
    [void]$ops.Add((Op-Set "/$sheet/A2:$lastCol$limit" @{ fill="FFF2CC"; "font.name"="Aptos"; "alignment.vertical"="top" }))
    [void]$ops.Add((Op-Set "/$sheet" @{ freeze="A2"; autoFilter="A1:$lastCol$limit"; orientation="landscape"; fitToPage="1x0"; showGridLines=$false }))
    Invoke-OfficeBatch $ops

    $ops = [System.Collections.ArrayList]::new()
    for ($i = 0; $i -lt $headers.Count; $i++) {
        $col = Get-ExcelColumn ($i + 1)
        $field = $headers[$i]
        $width = 14
        if ($field -match "_id$|version|category|status|source|decision|type|stage|basis") { $width = 18 }
        if ($field -match "path|file|notes|reason|protocol|sequence|condition|method|rule|action") { $width = 30 }
        if ($field -match "date|timestamp|_at$") { $width = 16 }
        [void]$ops.Add((Op-Set "/$sheet/col[$col]" @{ width=$width }))
    }
    [void]$ops.Add((Op-Set "/$sheet/row[1]" @{ height=42 }))
    Invoke-OfficeBatch $ops
}

# Formula columns
$ops = [System.Collections.ArrayList]::new()
for ($r = 2; $r -le 201; $r++) {
    [void]$ops.Add((Op-Set "/02_Recipe/M$r" @{ formula="IF(COUNTA(I${r}:L${r})=0,`"`",SUM(I${r}:L${r}))"; numberformat="0.00"; fill="F2F2F2" }))
    [void]$ops.Add((Op-Set "/02_Recipe/N$r" @{ formula="IF(M$r=`"`",`"`",IF(ABS(M$r-100)<=0.1,`"PASS`",`"CHECK`"))"; fill="F2F2F2" }))
    [void]$ops.Add((Op-Set "/08_Electrochem_Summary/K$r" @{ formula="IFERROR(I$r/E$r*100,`"`")"; numberformat="0.00"; fill="F2F2F2" }))
    [void]$ops.Add((Op-Set "/08_Electrochem_Summary/L$r" @{ formula="IFERROR(J$r/E$r*100,`"`")"; numberformat="0.00"; fill="F2F2F2" }))
    [void]$ops.Add((Op-Set "/08_Electrochem_Summary/P$r" @{ formula="IF(OR(E$r=`"`",J$r=`"`"),`"`",E$r-J$r)"; numberformat="0.00"; fill="F2F2F2" }))
}
for ($r = 2; $r -le 501; $r++) {
    [void]$ops.Add((Op-Set "/10_EDX_Measurements/M$r" @{ formula="IFERROR(H$r/I$r,`"`")"; numberformat="0.000"; fill="F2F2F2" }))
    [void]$ops.Add((Op-Set "/10_EDX_Measurements/N$r" @{ formula="IFERROR(J$r/K$r,`"`")"; numberformat="0.000"; fill="F2F2F2" }))
}
Invoke-OfficeBatch $ops

# Named ranges and validations
$ops = [System.Collections.ArrayList]::new()
$namedRanges = [ordered]@{
    "RecipeIDs"="'02_Recipe'!`$A`$2:`$A`$201"
    "MaterialLotIDs"="'03_Material_Lots'!`$A`$2:`$A`$201"
    "ElectrodeIDs"="'04_Electrode_Process'!`$A`$2:`$A`$201"
    "CellIDs"="'05_Cell_Assembly'!`$A`$2:`$A`$201"
    "ProtocolIDs"="'06_Test_Protocol'!`$A`$2:`$A`$201"
}
foreach ($entry in $namedRanges.GetEnumerator()) {
    [void]$ops.Add((Op-Add "/" "namedrange" @{ name=$entry.Key; ref=$entry.Value }))
}
Invoke-OfficeBatch $ops

$validationSpecs = @(
    @{S="02_Recipe"; R="C2:C201"; T="list"; F="='15_Lookups'!`$A`$2:`$A`$7"; P="Chọn trạng thái recipe."},
    @{S="02_Recipe"; R="D2:D201"; T="list"; F="='15_Lookups'!`$C`$2:`$C`$6"; P="Nguồn recipe."},
    @{S="02_Recipe"; R="E2:H201"; T="list"; F="=MaterialLotIDs"; P="Chọn material_lot_id đã khai báo."},
    @{S="02_Recipe"; R="I2:L201"; T="decimal"; F="0"; F2="100"; P="Nhập wt.% 0-100; tổng phải bằng 100."},
    @{S="02_Recipe"; R="R2:R201"; T="list"; F="='15_Lookups'!`$B`$2:`$B`$3"; P="Có được đưa vào không gian tối ưu?"},
    @{S="03_Material_Lots"; R="B2:B201"; T="list"; F="='15_Lookups'!`$D`$2:`$D`$8"; P="Vai trò vật liệu."},
    @{S="03_Material_Lots"; R="H2:H201"; T="list"; F="='15_Lookups'!`$M`$2:`$M`$6"; P="Trạng thái material lot."},
    @{S="04_Electrode_Process"; R="B2:B201"; T="list"; F="=RecipeIDs"; P="Recipe đã khai báo."},
    @{S="04_Electrode_Process"; R="G2:G201"; T="list"; F="='15_Lookups'!`$M`$2:`$M`$6"; P="Trạng thái chế tạo."},
    @{S="05_Cell_Assembly"; R="B2:B201"; T="list"; F="=ElectrodeIDs"; P="Electrode đã khai báo."},
    @{S="05_Cell_Assembly"; R="G2:G201"; T="list"; F="='15_Lookups'!`$M`$2:`$M`$6"; P="Trạng thái assembly."},
    @{S="05_Cell_Assembly"; R="H2:H201"; T="list"; F="='15_Lookups'!`$E`$2:`$E`$6"; P="Loại cell."},
    @{S="06_Test_Protocol"; R="B2:B201"; T="list"; F="=CellIDs"; P="Cell đã khai báo."},
    @{S="06_Test_Protocol"; R="D2:D201"; T="list"; F="='15_Lookups'!`$M`$2:`$M`$6"; P="Trạng thái protocol."},
    @{S="07_Cycle_Data"; R="B2:B1001"; T="list"; F="=CellIDs"; P="Cell đã khai báo."},
    @{S="07_Cycle_Data"; R="C2:C1001"; T="list"; F="=ProtocolIDs"; P="Protocol đã khai báo."},
    @{S="07_Cycle_Data"; R="E2:E1001"; T="list"; F="='15_Lookups'!`$N`$2:`$N`$6"; P="Loại bước đo."},
    @{S="07_Cycle_Data"; R="O2:O1001"; T="list"; F="='15_Lookups'!`$G`$2:`$G`$4"; P="QC cho dòng cycle."},
    @{S="08_Electrochem_Summary"; R="A2:A201"; T="list"; F="=CellIDs"; P="Cell đã khai báo."},
    @{S="08_Electrochem_Summary"; R="B2:B201"; T="list"; F="=RecipeIDs"; P="Recipe liên quan."},
    @{S="08_Electrochem_Summary"; R="C2:C201"; T="list"; F="=ElectrodeIDs"; P="Electrode liên quan."},
    @{S="08_Electrochem_Summary"; R="D2:D201"; T="list"; F="=ProtocolIDs"; P="Protocol liên quan."},
    @{S="08_Electrochem_Summary"; R="Q2:Q201"; T="list"; F="='15_Lookups'!`$H`$2:`$H`$5"; P="Phân biệt target đo thật và dự đoán sớm."},
    @{S="08_Electrochem_Summary"; R="T2:T201"; T="list"; F="='15_Lookups'!`$G`$2:`$G`$4"; P="QC summary."},
    @{S="09_SEM_Measurements"; R="B2:B501"; T="list"; F="=ElectrodeIDs"; P="Electrode đã khai báo."},
    @{S="09_SEM_Measurements"; R="C2:C501"; T="list"; F="=CellIDs"; P="Bỏ trống nếu ảnh pristine trước assembly."},
    @{S="09_SEM_Measurements"; R="D2:D501"; T="list"; F="='15_Lookups'!`$F`$2:`$F`$7"; P="Stage chụp ảnh."},
    @{S="09_SEM_Measurements"; R="W2:W501"; T="list"; F="='15_Lookups'!`$G`$2:`$G`$4"; P="QC ảnh/feature."},
    @{S="10_EDX_Measurements"; R="B2:B501"; T="list"; F="=ElectrodeIDs"; P="Electrode đã khai báo."},
    @{S="10_EDX_Measurements"; R="C2:C501"; T="list"; F="=CellIDs"; P="Cell nếu đo post-cycle."},
    @{S="10_EDX_Measurements"; R="D2:D501"; T="list"; F="='15_Lookups'!`$F`$2:`$F`$7"; P="Stage đo EDX."},
    @{S="10_EDX_Measurements"; R="G2:G501"; T="list"; F="='15_Lookups'!`$L`$2:`$L`$3"; P="Không trộn at.% và wt.%."},
    @{S="10_EDX_Measurements"; R="U2:U501"; T="list"; F="='15_Lookups'!`$G`$2:`$G`$4"; P="QC EDX."},
    @{S="11_Recommendation_Log"; R="V2:V201"; T="list"; F="='15_Lookups'!`$I`$2:`$I`$5"; P="Exploit/explore/control/boundary."},
    @{S="11_Recommendation_Log"; R="X2:X201"; T="list"; F="='15_Lookups'!`$J`$2:`$J`$5"; P="Quyết định human-in-the-loop."},
    @{S="12_Validation_Log"; R="C2:C201"; T="list"; F="='15_Lookups'!`$O`$2:`$O`$5"; P="Kiểu validation."},
    @{S="12_Validation_Log"; R="W2:W201"; T="list"; F="='15_Lookups'!`$G`$2:`$G`$4"; P="Kết luận validation."},
    @{S="13_Failure_Log"; R="F2:F201"; T="list"; F="='15_Lookups'!`$K`$2:`$K`$7"; P="Loại failure."},
    @{S="13_Failure_Log"; R="H2:I201"; T="list"; F="='15_Lookups'!`$B`$2:`$B`$3"; P="Yes/No."}
)
$ops = [System.Collections.ArrayList]::new()
foreach ($v in $validationSpecs) {
    $props = @{ ref=$v.R; type=$v.T; formula1=$v.F; allowBlank=$true; showError=$true; showInput=$true; errorTitle="Dữ liệu không hợp lệ"; error="Kiểm tra data dictionary và danh mục chuẩn."; promptTitle="Quy ước nhập"; prompt=$v.P }
    if ($v.ContainsKey("F2")) { $props.formula2 = $v.F2; $props.operator = "between" }
    [void]$ops.Add((Op-Add "/$($v.S)" "validation" $props))
}
Invoke-OfficeBatch $ops

# Conditional formatting for status/QC columns
$cfSpecs = @(
    @{S="02_Recipe"; R="N2:N201"}, @{S="07_Cycle_Data"; R="O2:O1001"}, @{S="08_Electrochem_Summary"; R="T2:T201"},
    @{S="09_SEM_Measurements"; R="W2:W501"}, @{S="10_EDX_Measurements"; R="U2:U501"}, @{S="12_Validation_Log"; R="W2:W201"}
)
$ops = [System.Collections.ArrayList]::new()
foreach ($cf in $cfSpecs) {
    [void]$ops.Add((Op-Add "/$($cf.S)" "conditionalformatting" @{ type="containsText"; ref=$cf.R; text="PASS"; fill="C6EFCE" }))
    [void]$ops.Add((Op-Add "/$($cf.S)" "conditionalformatting" @{ type="containsText"; ref=$cf.R; text="CHECK"; fill="FFEB9C" }))
    [void]$ops.Add((Op-Add "/$($cf.S)" "conditionalformatting" @{ type="containsText"; ref=$cf.R; text="FAIL"; fill="FFC7CE" }))
}
Invoke-OfficeBatch $ops

# Data dictionary generated from schemas
$dictHeaders = @("sheet_name","field_name","data_stage","role","priority","known_before_experiment","unit_or_type","qc_rule")
$ops = [System.Collections.ArrayList]::new()
for ($i = 0; $i -lt $dictHeaders.Count; $i++) {
    $col = Get-ExcelColumn ($i + 1)
    [void]$ops.Add((Op-Set "/01_Danh_muc_truong/$($col)1" @{ value=$dictHeaders[$i]; type="string" }))
}
$dictRow = 2
foreach ($sheet in $schemas.Keys) {
    $stage = if ($sheet -match "Recipe|Material|Electrode|Cell|Protocol") { "Pre-experiment / controllable" } elseif ($sheet -match "Recommendation") { "Decision" } elseif ($sheet -match "Validation|Failure") { "Governance" } else { "Post-experiment / measured" }
    foreach ($field in $schemas[$sheet]) {
        $known = if ($stage -eq "Pre-experiment / controllable") { "Yes" } elseif ($stage -eq "Decision") { "Model output" } else { "No / derived" }
        $values = @($sheet,$field,$stage,(Get-Role $sheet $field),(Get-Priority $field),$known,(Get-Unit $field),(Get-Rule $field))
        for ($i = 0; $i -lt $values.Count; $i++) {
            $col = Get-ExcelColumn ($i + 1)
            [void]$ops.Add((Op-Set "/01_Danh_muc_truong/$col$dictRow" @{ value=[string]$values[$i]; type="string" }))
        }
        $dictRow++
    }
}
Invoke-OfficeBatch $ops

$ops = [System.Collections.ArrayList]::new()
[void]$ops.Add((Op-Set "/01_Danh_muc_truong/A1:H1" @{ fill="1F4E78"; "font.color"="FFFFFF"; "font.bold"=$true; "alignment.wrapText"=$true; "alignment.vertical"="center"; border="thin"; "border.color"="D9E2F3" }))
[void]$ops.Add((Op-Set "/01_Danh_muc_truong/A2:H$($dictRow-1)" @{ "font.name"="Aptos"; "alignment.wrapText"=$true; "alignment.vertical"="top"; "border.bottom"="thin"; "border.color"="D9E2F3" }))
[void]$ops.Add((Op-Set "/01_Danh_muc_truong" @{ freeze="A2"; autoFilter="A1:H$($dictRow-1)"; orientation="landscape"; fitToPage="1x0"; showGridLines=$false }))
foreach ($spec in @(@("A",22),@("B",32),@("C",24),@("D",24),@("E",10),@("F",22),@("G",16),@("H",42))) {
    [void]$ops.Add((Op-Set "/01_Danh_muc_truong/col[$($spec[0])]" @{ width=$spec[1] }))
}
[void]$ops.Add((Op-Set "/01_Danh_muc_truong/row[1]" @{ height=36 }))
Invoke-OfficeBatch $ops

# QC dashboard
$qcRows = @(
    @("Số recipe","COUNTA('02_Recipe'!A2:A201)",">= 20 trước model V1","Độ phủ design space"),
    @("Số electrode","COUNTA('04_Electrode_Process'!A2:A201)",">= recipe","Có replicate/batch"),
    @("Số cell","COUNTA('05_Cell_Assembly'!A2:A201)",">= electrode","Nên có replicate cho control/top candidate"),
    @("Số protocol","COUNTA('06_Test_Protocol'!A2:A201)","= cell","Không trộn protocol không ghi nhận"),
    @("Số electrochem summary","COUNTA('08_Electrochem_Summary'!A2:A201)","= cell đã test","Coverage outcome"),
    @("Target observed","COUNTIF('08_Electrochem_Summary'!Q2:Q201,`"observed`")","> 0","Dùng cho target thật"),
    @("Target early-predicted","COUNTIF('08_Electrochem_Summary'!Q2:Q201,`"early_predicted`")","Tách khỏi observed","Không đánh đồng với ground truth"),
    @("SEM measurements","COUNTA('09_SEM_Measurements'!A2:A501)","Theo ROI/stage","Có metadata ảnh"),
    @("EDX measurements","COUNTA('10_EDX_Measurements'!A2:A501)","Theo ROI/stage","Cùng basis"),
    @("Recommendations logged","COUNTA('11_Recommendation_Log'!A2:A201)","> 0 khi closed-loop","Ghi trước kết quả"),
    @("Validation runs","COUNTA('12_Validation_Log'!A2:A201)",">= 1 prospective","Bằng chứng thi"),
    @("Failures logged","COUNTA('13_Failure_Log'!A2:A201)","Không xóa failure","Chống selection bias"),
    @("Composition CHECK","COUNTIF('02_Recipe'!N2:N201,`"CHECK`")","= 0","Tổng thành phần phải bằng 100"),
    @("Cell-summary coverage","IF(B6=0,0,B8/B6)",">= 80%","Tỷ lệ cell có summary")
)
$ops = [System.Collections.ArrayList]::new()
[void]$ops.Add((Op-Set "/14_QC_Dashboard/A1" @{ value="QC DASHBOARD — DATA READINESS FOR CLOSED-LOOP AI"; type="string"; merge="A1:D1"; fill="17365D"; "font.color"="FFFFFF"; "font.bold"=$true; "font.size"="16pt"; "alignment.vertical"="center" }))
$headers = @("Kiểm tra","Kết quả","Yêu cầu","Ý nghĩa")
for ($i=0; $i -lt 4; $i++) { $col=Get-ExcelColumn($i+1); [void]$ops.Add((Op-Set "/14_QC_Dashboard/$($col)3" @{ value=$headers[$i]; type="string" })) }
for ($i=0; $i -lt $qcRows.Count; $i++) {
    $r=$i+4
    [void]$ops.Add((Op-Set "/14_QC_Dashboard/A$r" @{ value=$qcRows[$i][0]; type="string" }))
    [void]$ops.Add((Op-Set "/14_QC_Dashboard/B$r" @{ formula=$qcRows[$i][1]; fill="F2F2F2" }))
    [void]$ops.Add((Op-Set "/14_QC_Dashboard/C$r" @{ value=$qcRows[$i][2]; type="string" }))
    [void]$ops.Add((Op-Set "/14_QC_Dashboard/D$r" @{ value=$qcRows[$i][3]; type="string" }))
}
[void]$ops.Add((Op-Set "/14_QC_Dashboard/A3:D3" @{ fill="1F4E78"; "font.color"="FFFFFF"; "font.bold"=$true; "alignment.wrapText"=$true; "alignment.vertical"="center" }))
[void]$ops.Add((Op-Set "/14_QC_Dashboard/A4:D$($qcRows.Count+3)" @{ "alignment.wrapText"=$true; "alignment.vertical"="top"; "border.bottom"="thin"; "border.color"="D9E2F3" }))
[void]$ops.Add((Op-Set "/14_QC_Dashboard/B17" @{ numberformat="0.0%" }))
[void]$ops.Add((Op-Set "/14_QC_Dashboard" @{ freeze="A4"; orientation="landscape"; fitToPage=$true; showGridLines=$false }))
foreach ($spec in @(@("A",28),@("B",16),@("C",24),@("D",44))) { [void]$ops.Add((Op-Set "/14_QC_Dashboard/col[$($spec[0])]" @{ width=$spec[1] })) }
Invoke-OfficeBatch $ops

# Guide, lookups formatting
$ops = [System.Collections.ArrayList]::new()
[void]$ops.Add((Op-Set "/00_Huong_dan/A1" @{ merge="A1:H1"; fill="17365D"; "font.color"="FFFFFF"; "font.bold"=$true; "font.size"="16pt"; "alignment.vertical"="center" }))
foreach ($r in @(3,5,7,8,9,10,12,14)) { [void]$ops.Add((Op-Set "/00_Huong_dan/A$r" @{ fill="0F766E"; "font.color"="FFFFFF"; "font.bold"=$true; "alignment.wrapText"=$true; "alignment.vertical"="center" })) }
[void]$ops.Add((Op-Set "/00_Huong_dan/A16:B16" @{ fill="1F4E78"; "font.color"="FFFFFF"; "font.bold"=$true; "alignment.vertical"="center" }))
[void]$ops.Add((Op-Set "/00_Huong_dan/B3:H14" @{ "alignment.wrapText"=$true; "alignment.vertical"="top" }))
[void]$ops.Add((Op-Set "/00_Huong_dan/A17:B30" @{ "alignment.wrapText"=$true; "alignment.vertical"="top"; "border.bottom"="thin"; "border.color"="D9E2F3" }))
[void]$ops.Add((Op-Set "/00_Huong_dan/col[A]" @{ width=28 }))
[void]$ops.Add((Op-Set "/00_Huong_dan/col[B]" @{ width=70 }))
[void]$ops.Add((Op-Set "/00_Huong_dan" @{ freeze="A2"; orientation="landscape"; fitToPage=$true; showGridLines=$false }))
$lastLookupCol = Get-ExcelColumn $lookups.Count
[void]$ops.Add((Op-Set "/15_Lookups/A1:$($lastLookupCol)1" @{ fill="1F4E78"; "font.color"="FFFFFF"; "font.bold"=$true; "alignment.wrapText"=$true }))
[void]$ops.Add((Op-Set "/15_Lookups/A2:$($lastLookupCol)20" @{ fill="F2F2F2" }))
[void]$ops.Add((Op-Set "/15_Lookups" @{ freeze="A2"; showGridLines=$false }))
for ($i=1; $i -le $lookups.Count; $i++) { $col=Get-ExcelColumn $i; [void]$ops.Add((Op-Set "/15_Lookups/col[$col]" @{ width=22 })) }
Invoke-OfficeBatch $ops

# Number formats for common ranges
$formatSpecs = @(
    @{S="02_Recipe"; R="I2:M201"; F="0.00"},
    @{S="04_Electrode_Process"; R="I2:AC201"; F="0.00"},
    @{S="05_Cell_Assembly"; R="N2:R201"; F="0.00"},
    @{S="06_Test_Protocol"; R="I2:T201"; F="0.00"},
    @{S="07_Cycle_Data"; R="D2:L1001"; F="0.000"},
    @{S="08_Electrochem_Summary"; R="E2:S201"; F="0.00"},
    @{S="09_SEM_Measurements"; R="H2:R501"; F="0.000"},
    @{S="10_EDX_Measurements"; R="H2:P501"; F="0.000"},
    @{S="11_Recommendation_Log"; R="H2:U201"; F="0.00"},
    @{S="12_Validation_Log"; R="E2:R201"; F="0.000"}
)
$ops = [System.Collections.ArrayList]::new()
foreach ($fmt in $formatSpecs) { [void]$ops.Add((Op-Set "/$($fmt.S)/$($fmt.R)" @{ numberformat=$fmt.F })) }
Invoke-OfficeBatch $ops

officecli save $OutputFile | Out-Null
officecli close $OutputFile | Out-Null

Write-Output $OutputFile
