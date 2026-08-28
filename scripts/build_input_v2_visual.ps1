$ErrorActionPreference = "Stop"
$File = "F:\AI\GTIP\outputs\Mau_nhap_du_lieu_vat_lieu_cho_AI_v2_viewer.xlsx"

function Get-Col([int]$n) {
    $s = ""
    while ($n -gt 0) { $n--; $s = [char](65 + ($n % 26)) + $s; $n = [math]::Floor($n / 26) }
    $s
}
function To-AsciiJson($value) {
    $json = $value | ConvertTo-Json -Depth 30 -Compress
    $sb = New-Object System.Text.StringBuilder
    foreach ($ch in $json.ToCharArray()) {
        $code = [int][char]$ch
        if ($code -gt 127) { [void]$sb.Append(('\u{0:x4}' -f $code)) } else { [void]$sb.Append($ch) }
    }
    $sb.ToString()
}
function Batch([System.Collections.ArrayList]$ops) {
    for ($i=0; $i -lt $ops.Count; $i+=45) {
        $end=[math]::Min($i+44,$ops.Count-1); $chunk=@($ops[$i..$end]); $ascii=To-AsciiJson $chunk
        $out=$ascii | officecli batch $File --json
        if ($LASTEXITCODE -ne 0) { throw "officecli batch failed: $out" }
    }
}
function S($ops,[string]$path,[hashtable]$props){ [void]$ops.Add(@{command="set";path=$path;props=$props}) }
function V($ops,[string]$sheet,[string]$ref,[string]$type,[string]$f1,[string]$prompt,[string]$f2=""){
    $p=@{ref=$ref;type=$type;formula1=$f1;allowBlank=$false;showError=$true;showInput=$true;errorTitle="Giá trị chưa hợp lệ";error="Kiểm tra đơn vị hoặc chọn trong danh sách.";promptTitle="Cách nhập";prompt=$prompt}
    if($f2){$p.formula2=$f2;$p.operator="between"}
    [void]$ops.Add(@{command="add";parent="/$sheet";type="validation";props=$p})
}

# Reuse the known viewer-compatible V1 package without deleting worksheet parts.
$ops=New-Object System.Collections.ArrayList
S $ops "/01_Nhap_cong_thuc" @{name="01_Nhap_cong_thuc"}
S $ops "/02_Ket_qua_SEM_EDX" @{name="02_Ket_qua_SEM_EDX"}
S $ops "/03_Ket_qua_dien_hoa" @{name="03_Ket_qua_dien_hoa"}
S $ops "/04_Phan_hoi_AI" @{name="04_Phan_hoi_AI"}
S $ops "/06_QC_tom_tat" @{name="06_QC_tom_tat"}
S $ops "/06_Mapping_AI" @{name="06_Mapping_AI";hidden=$true}
S $ops "/07_Backend_1" @{name="07_Backend_1";hidden=$true}
S $ops "/08_Backend_2" @{name="08_Backend_2";hidden=$true}
Batch $ops

$fields=[ordered]@{
"01_Nhap_cong_thuc"=@(
 @("Mã mẫu","sample_id","Team","Text","Mã duy nhất, ví dụ S001."),
 @("Mã công thức","recipe_id","Team","Text","Mẫu lặp cùng công thức dùng chung mã."),
 @("Mã lô thí nghiệm","batch_id","Team","Text","Mã mẻ/ngày làm chung."),
 @("Lần lặp","replicate_no","Team","Số nguyên","1, 2, 3..."),
 @("Ngày làm","experiment_date","Team","Ngày","Ngày chế tạo điện cực."),
 @("Người thực hiện","operator","Team","Text","Tên hoặc mã người làm."),
 @("Si (wt.%)","si_content","Team","wt.%","Phần trăm khối lượng nano-Si."),
 @("MXene (wt.%)","mxene_content","Team","wt.%","Phần trăm khối lượng MXene."),
 @("Alginate (wt.%)","alginate_content","Team","wt.%","Phần trăm binder alginate."),
 @("Carbon (wt.%)","carbon_content","Team","wt.%","Phần trăm carbon dẫn điện."),
 @("Tổng (%)","composition_sum","Công thức","%","Tự tính; phải bằng 100 ± 0,1."),
 @("Kiểm tra tổng","composition_qc","Công thức","PASS/CHECK","Tự tính."),
 @("Thời gian trộn (phút)","mixing_time","Team","phút","Tổng thời gian trộn thực tế."),
 @("Nhiệt độ sấy (°C)","drying_temp","Team","°C","Nhiệt độ setpoint."),
 @("Áp lực ép (MPa)","pressing_pressure","Team","MPa","Luôn dùng MPa."),
 @("Khối lượng phủ (mg/cm²)","mass_loading","Team","mg/cm²","Cần để so sánh điện hóa công bằng."),
 @("Trạng thái","experiment_status","Team","Danh mục","planned/in_progress/completed/failed.")
);
"02_Ket_qua_SEM_EDX"=@(
 @("Mã mẫu","sample_id","Team","Text","Trùng sheet công thức."),
 @("Giai đoạn đo","measurement_stage","Team","Danh mục","Thời điểm chụp/đo."),
 @("Kích thước hạt TB (nm)","particle_size_mean","Team","nm","Giá trị trung bình từ SEM."),
 @("Điểm độ xốp (0–1)","porosity_score","Team/Phần mềm","0–1","Dùng cùng cách tính."),
 @("Chỉ số kết tụ (0–1)","agglomeration_index","Team/Phần mềm","0–1","Dùng cùng cách tính."),
 @("Mật độ vết nứt (0–1)","crack_density","Team/Phần mềm","0–1","Dùng cùng cách tính."),
 @("Độ đồng đều (0–1)","surface_uniformity","Team/Phần mềm","0–1","Dùng cùng cách tính."),
 @("Basis EDX","edx_basis","Team","at.%/wt.%","Chọn đúng basis máy xuất."),
 @("Si (%)","si_percent","Team","%","Kết quả EDX."),
 @("Ti (%)","ti_percent","Team","%","Kết quả EDX."),
 @("C (%)","c_percent","Team","%","Kết quả EDX."),
 @("O (%)","o_percent","Team","%","Kết quả EDX."),
 @("Tạp chất (%)","impurity_percent","Team","%","Nguyên tố khác Si/Ti/C/O."),
 @("Tỷ số Si/Ti","si_ti_ratio","Công thức","ratio","Tự tính."),
 @("Tỷ số C/O","c_o_ratio","Công thức","ratio","Tự tính."),
 @("Trạng thái QC","qc_status","Team","PASS/CHECK/FAIL","Chất lượng phép đo.")
);
"03_Ket_qua_dien_hoa"=@(
 @("Mã mẫu","sample_id","Team","Text","Trùng sheet công thức."),
 @("Tên protocol","protocol_name","Team","Text","Tên ngắn, nhất quán."),
 @("C-rate","c_rate","Team","C","Ví dụ 0.1, 0.5, 1."),
 @("Điện áp thấp (V)","voltage_min","Team","V","Cận dưới voltage window."),
 @("Điện áp cao (V)","voltage_max","Team","V","Cận trên voltage window."),
 @("Nhiệt độ test (°C)","test_temp","Team","°C","Nhiệt độ test."),
 @("Dung lượng ban đầu","initial_capacity","Team","mAh/g","Cùng quy ước chu kỳ."),
 @("Dung lượng chu kỳ 50","capacity_50","Team","mAh/g","Tại cycle 50."),
 @("Dung lượng chu kỳ 100","capacity_100","Team","mAh/g","Tại cycle 100."),
 @("Retention 50 (%)","retention_50","Công thức","%","Tự tính."),
 @("Retention 100 (%)","retention_100","Công thức","%","Target chính."),
 @("Suy giảm dung lượng","capacity_fade","Công thức","mAh/g","Tự tính."),
 @("Hiệu suất Coulomb (%)","coulombic_efficiency","Team","%","Theo cùng quy ước."),
 @("Rct (Ω)","rct","Team","Ω","Cùng equivalent circuit."),
 @("Trạng thái test","test_status","Team","Danh mục","completed/failed/censored.")
);
"04_Phan_hoi_AI"=@(
 @("Mã gợi ý","recommendation_id","AI","Text","AI tự ghi."),
 @("Công thức được gợi ý","suggested_recipe","AI","Text","AI tự ghi."),
 @("Retention dự đoán (%)","predicted_retention_100","AI","%","AI tự ghi."),
 @("Độ bất định","uncertainty_std","AI","%","AI tự ghi."),
 @("Lý do chọn","selection_reason","AI","Danh mục","AI tự ghi."),
 @("Quyết định","reviewer_decision","Team","Danh mục","Team chọn accept/modify/reject/defer."),
 @("Mã mẫu đã làm","actual_sample_id","Team","Text","Liên kết gợi ý với mẫu thực tế."),
 @("Phiên bản model","model_version","AI","Text","Cột ẩn; AI tự ghi."),
 @("Ngày chốt train","training_cutoff","AI","Ngày giờ","Cột ẩn; AI tự ghi."),
 @("Số mẫu train","training_sample_count","AI","Số nguyên","Cột ẩn; AI tự ghi."),
 @("Hàm chọn mẫu","acquisition_function","AI","Text","Cột ẩn; AI tự ghi."),
 @("Điểm chọn mẫu","acquisition_score","AI","Số","Cột ẩn; AI tự ghi.")
)
}
$limits=@{"01_Nhap_cong_thuc"=201;"02_Ket_qua_SEM_EDX"=301;"03_Ket_qua_dien_hoa"=201;"04_Phan_hoi_AI"=201}
$tab=@{"01_Nhap_cong_thuc"="2F75B5";"02_Ket_qua_SEM_EDX"="ED7D31";"03_Ket_qua_dien_hoa"="70AD47";"04_Phan_hoi_AI"="7030A0"}

# Headers, widths and V1-style input areas.
$ops=New-Object System.Collections.ArrayList
foreach($sheet in $fields.Keys){
 $defs=$fields[$sheet];$last=Get-Col $defs.Count
 for($i=0;$i -lt $defs.Count;$i++){ $c=Get-Col($i+1); S $ops "/$sheet/$($c)1" @{value=$defs[$i][0];type="string"}; S $ops "/$sheet/col[$c]" @{width=([math]::Max(12,[math]::Min(24,$defs[$i][0].Length+3)))} }
 S $ops "/$sheet/A1:$($last)1" @{fill="1F4E78";"font.color"="FFFFFF";"font.bold"=$true;"font.name"="Times New Roman";"font.size"="11pt";"alignment.wrapText"=$true;"alignment.vertical"="center";border="thin";"border.color"="D9E2F3"}
 S $ops "/$sheet/row[1]" @{height=44}
 S $ops "/$sheet/A2:$last$($limits[$sheet])" @{fill="FFF2CC";"font.name"="Calibri";"alignment.vertical"="top";"border.bottom"="thin";"border.color"="E7E6E6"}
 S $ops "/$sheet" @{freeze="B2";autoFilter="A1:$last$($limits[$sheet])";orientation="landscape";fitToPage="1x0";showGridLines=$false}
}
S $ops "/01_Nhap_cong_thuc/K2:L201" @{fill="F2F2F2"}
S $ops "/02_Ket_qua_SEM_EDX/N2:O301" @{fill="F2F2F2"}
S $ops "/03_Ket_qua_dien_hoa/J2:L201" @{fill="F2F2F2"}
S $ops "/04_Phan_hoi_AI/A2:E201" @{fill="E4DFEC"}
S $ops "/04_Phan_hoi_AI/F2:G201" @{fill="E2F0D9"}
foreach($c in @("H","I","J","K","L")){ S $ops "/04_Phan_hoi_AI/col[$c]" @{hidden=$true} }
Batch $ops

# Formulas.
$ops=New-Object System.Collections.ArrayList
for($r=2;$r -le 201;$r++){
 S $ops "/01_Nhap_cong_thuc/K$r" @{formula="IF(COUNTA(G${r}:J${r})=0,`"`",SUM(G${r}:J${r}))";numberformat="0.00";fill="F2F2F2"}
 S $ops "/01_Nhap_cong_thuc/L$r" @{formula="IF(K$r=`"`",`"`",IF(ABS(K$r-100)<=0.1,`"PASS`",`"CHECK`"))";fill="F2F2F2"}
 S $ops "/03_Ket_qua_dien_hoa/J$r" @{formula="IFERROR(H$r/G$r*100,`"`")";numberformat="0.00";fill="F2F2F2"}
 S $ops "/03_Ket_qua_dien_hoa/K$r" @{formula="IFERROR(I$r/G$r*100,`"`")";numberformat="0.00";fill="F2F2F2"}
 S $ops "/03_Ket_qua_dien_hoa/L$r" @{formula="IF(OR(G$r=`"`",I$r=`"`"),`"`",G$r-I$r)";numberformat="0.00";fill="F2F2F2"}
}
for($r=2;$r -le 301;$r++){
 S $ops "/02_Ket_qua_SEM_EDX/N$r" @{formula="IFERROR(I$r/J$r,`"`")";numberformat="0.000";fill="F2F2F2"}
 S $ops "/02_Ket_qua_SEM_EDX/O$r" @{formula="IFERROR(K$r/L$r,`"`")";numberformat="0.000";fill="F2F2F2"}
}
Batch $ops

# Validations only for fields that can be misunderstood.
$ops=New-Object System.Collections.ArrayList
V $ops "01_Nhap_cong_thuc" "D2:D201" "whole" "1" "Nhập lần lặp từ 1 đến 20." "20"
V $ops "01_Nhap_cong_thuc" "G2:J201" "decimal" "0" "Nhập wt.% 0–100; tổng bốn thành phần bằng 100." "100"
V $ops "01_Nhap_cong_thuc" "Q2:Q201" "list" "planned,in_progress,completed,failed" "Chọn trạng thái."
V $ops "02_Ket_qua_SEM_EDX" "B2:B301" "list" "pristine,post_formation,post_cycle_10,post_cycle_50,post_cycle_100" "Chọn đúng thời điểm đo."
V $ops "02_Ket_qua_SEM_EDX" "D2:G301" "decimal" "0" "Nhập chỉ số từ 0 đến 1." "1"
V $ops "02_Ket_qua_SEM_EDX" "H2:H301" "list" "at.%,wt.%" "Chọn basis máy EDX xuất."
V $ops "02_Ket_qua_SEM_EDX" "I2:M301" "decimal" "0" "Nhập phần trăm 0–100." "100"
V $ops "02_Ket_qua_SEM_EDX" "P2:P301" "list" "PASS,CHECK,FAIL" "Chọn chất lượng phép đo."
V $ops "03_Ket_qua_dien_hoa" "M2:M201" "decimal" "0" "Nhập hiệu suất Coulomb 0–100." "100"
V $ops "03_Ket_qua_dien_hoa" "O2:O201" "list" "completed,failed,censored" "Chọn trạng thái test."
V $ops "04_Phan_hoi_AI" "E2:E201" "list" "exploit,explore,control,boundary" "AI tự ghi."
V $ops "04_Phan_hoi_AI" "F2:F201" "list" "accept,modify,reject,defer" "Team chọn quyết định."
Batch $ops

# Guide kept as direct as V1.
$ops=New-Object System.Collections.ArrayList
S $ops "/00_Huong_dan/A1" @{value="MẪU NHẬP DỮ LIỆU VẬT LIỆU V2 — CHỈ GIỮ TRƯỜNG BẮT BUỘC";type="string";merge="A1:F1";fill="1F4E78";"font.color"="FFFFFF";"font.bold"=$true;"font.size"="16pt";"font.name"="Times New Roman";"alignment.vertical"="center"}
S $ops "/00_Huong_dan/A3" @{value="Mục đích";type="string";fill="0F766E";"font.color"="FFFFFF";"font.bold"=$true;"alignment.horizontal"="center"}
S $ops "/00_Huong_dan/B3" @{value="Team chỉ nhập giá trị mình trực tiếp làm hoặc đọc từ máy; công thức và metadata AI được tự tính hoặc ẩn.";type="string";merge="B3:F3";"alignment.wrapText"=$true}
S $ops "/00_Huong_dan/A5" @{value="Cách dùng nhanh";type="string";merge="A5:F5";fill="1F4E78";"font.color"="FFFFFF";"font.bold"=$true;"alignment.horizontal"="center"}
$steps=@(
"1. 01_Nhap_cong_thuc: nhập công thức và điều kiện chế tạo của từng mẫu.",
"2. 02_Ket_qua_SEM_EDX: nhập đúng các chỉ số SEM/EDX đã đo; không đo thì chưa tạo dòng.",
"3. 03_Ket_qua_dien_hoa: nhập kết quả test; retention được tự tính ở ô xám.",
"4. 04_Phan_hoi_AI: AI điền phần tím; team chỉ chọn Quyết định và Mã mẫu đã làm ở ô xanh.",
"5. Dùng cùng một Mã mẫu trên ba sheet dữ liệu. Không xóa mẫu failed/censored."
)
for($i=0;$i -lt $steps.Count;$i++){S $ops "/00_Huong_dan/A$($i+6)" @{value=$steps[$i];type="string";merge="A$($i+6):F$($i+6)";"alignment.wrapText"=$true}}
S $ops "/00_Huong_dan/A12" @{value="Màu ô";type="string";merge="A12:F12";fill="1F4E78";"font.color"="FFFFFF";"font.bold"=$true;"alignment.horizontal"="center"}
S $ops "/00_Huong_dan/A13" @{value="VÀNG — Team nhập";type="string";merge="A13:B13";fill="FFF2CC";"font.bold"=$true;"alignment.horizontal"="center"}
S $ops "/00_Huong_dan/C13" @{value="XÁM/TÍM — Công thức hoặc AI tự ghi";type="string";merge="C13:D13";fill="F2F2F2";"font.bold"=$true;"alignment.horizontal"="center"}
S $ops "/00_Huong_dan/E13" @{value="XANH — Team phản hồi AI";type="string";merge="E13:F13";fill="E2F0D9";"font.bold"=$true;"alignment.horizontal"="center"}
S $ops "/00_Huong_dan/A15" @{value="Quy tắc quan trọng";type="string";merge="A15:F15";fill="1F4E78";"font.color"="FFFFFF";"font.bold"=$true;"alignment.horizontal"="center"}
$rules=@("Không đoán số liệu và không điền 0 thay cho thiếu dữ liệu.","Bốn thành phần phải có tổng bằng 100 ± 0,1.","SEM/EDX là kết quả sau chế tạo; AI không dùng chúng để đề xuất recipe chưa làm.","Giữ lại thí nghiệm thất bại để tránh model chỉ học các mẫu đẹp.")
for($i=0;$i -lt $rules.Count;$i++){S $ops "/00_Huong_dan/A$($i+16)" @{value=($i+1).ToString()+". "+$rules[$i];type="string";merge="A$($i+16):F$($i+16)";"alignment.wrapText"=$true}}
S $ops "/00_Huong_dan" @{freeze="A5";showGridLines=$false;orientation="landscape";fitToPage=$true}
S $ops "/00_Huong_dan/col[A]" @{width=26};foreach($c in @("B","C","D","E","F")){S $ops "/00_Huong_dan/col[$c]" @{width=22}}
foreach($r in @(1,5,12,15)){S $ops "/00_Huong_dan/row[$r]" @{height=32}}
foreach($r in @(6,7,8,9,10,16,17,18,19)){S $ops "/00_Huong_dan/row[$r]" @{height=26}}
Batch $ops

# QC summary.
$ops=New-Object System.Collections.ArrayList
S $ops "/06_QC_tom_tat/A1" @{value="KIỂM TRA NHANH TRƯỚC KHI ĐƯA VÀO AI";type="string";merge="A1:D1";fill="1F4E78";"font.color"="FFFFFF";"font.bold"=$true;"font.size"="16pt";"font.name"="Times New Roman"}
$hh=@("Kiểm tra","Kết quả","Yêu cầu","Cách xử lý");for($i=0;$i -lt 4;$i++){S $ops "/06_QC_tom_tat/$(Get-Col($i+1))3" @{value=$hh[$i];type="string"}}
$qr=@(
@("Số mẫu công thức","COUNTA('01_Nhap_cong_thuc'!A2:A201)",">= 20","Tiếp tục phủ design space."),
@("Mẫu sai tổng thành phần","COUNTIF('01_Nhap_cong_thuc'!L2:L201,`"CHECK`")","= 0","Sửa tổng bốn thành phần."),
@("Mẫu có SEM/EDX","COUNTA('02_Ket_qua_SEM_EDX'!A2:A301)","Theo kế hoạch đo","Không tạo dòng nếu chưa đo."),
@("Mẫu có điện hóa","COUNTA('03_Ket_qua_dien_hoa'!A2:A201)",">= 15","Cần outcome thật để train."),
@("Target retention 100","COUNT('03_Ket_qua_dien_hoa'!K2:K201)",">= 15","Ưu tiên hoàn tất cycle 100."),
@("Phản hồi gợi ý AI","COUNTA('04_Phan_hoi_AI'!F2:F201)","> 0 khi closed-loop","Ghi quyết định trước khi làm mẫu.")
)
for($i=0;$i -lt $qr.Count;$i++){ $r=$i+4;S $ops "/06_QC_tom_tat/A$r" @{value=$qr[$i][0];type="string"};S $ops "/06_QC_tom_tat/B$r" @{formula=$qr[$i][1];fill="F2F2F2";numberformat="#,##0"};S $ops "/06_QC_tom_tat/C$r" @{value=$qr[$i][2];type="string"};S $ops "/06_QC_tom_tat/D$r" @{value=$qr[$i][3];type="string"}}
S $ops "/06_QC_tom_tat/A11" @{value="Trạng thái dataset";type="string";fill="D9EAD3";"font.bold"=$true}
S $ops "/06_QC_tom_tat/B11" @{formula="IF(AND(B4>=20,B5=0,B7>=15,B8>=15),`"SẴN SÀNG MODEL BASELINE`",`"CẦN BỔ SUNG DỮ LIỆU`")";fill="D9EAD3";"font.bold"=$true}
S $ops "/06_QC_tom_tat/C11:D11" @{fill="D9EAD3"}
S $ops "/06_QC_tom_tat/A3:D3" @{fill="1F4E78";"font.color"="FFFFFF";"font.bold"=$true;"font.name"="Times New Roman";"alignment.horizontal"="center";border="thin";"border.color"="D9E2F3"}
S $ops "/06_QC_tom_tat/A4:D11" @{"font.name"="Times New Roman";"font.size"="11pt";"alignment.wrapText"=$true;"alignment.vertical"="center";"border.bottom"="thin";"border.color"="D9E2F3"}
foreach($x in @(@("A",30),@("B",24),@("C",24),@("D",48))){S $ops "/06_QC_tom_tat/col[$($x[0])]" @{width=$x[1]}}
foreach($r in 4..11){S $ops "/06_QC_tom_tat/row[$r]" @{height=34}}
S $ops "/06_QC_tom_tat" @{freeze="A4";showGridLines=$false;orientation="landscape";fitToPage=$true}
Batch $ops

# Hidden technical mapping: keeps Vietnamese headers compatible with the AI pipeline.
$ops=New-Object System.Collections.ArrayList
$mh=@("sheet","header_vi","machine_key","filled_by","unit")
for($i=0;$i -lt $mh.Count;$i++){S $ops "/06_Mapping_AI/$(Get-Col($i+1))1" @{value=$mh[$i];type="string"}}
$row=2
foreach($sheet in $fields.Keys){foreach($d in $fields[$sheet]){$vals=@($sheet,$d[0],$d[1],$d[2],$d[3]);for($i=0;$i -lt 5;$i++){S $ops "/06_Mapping_AI/$(Get-Col($i+1))$row" @{value=[string]$vals[$i];type="string"}};$row++}}
S $ops "/06_Mapping_AI/A1:E1" @{fill="1F4E78";"font.color"="FFFFFF";"font.bold"=$true}
S $ops "/06_Mapping_AI" @{hidden=$true}
Batch $ops

# Conditional formatting.
$ops=New-Object System.Collections.ArrayList
[void]$ops.Add(@{command="add";parent="/01_Nhap_cong_thuc";type="conditionalformatting";props=@{ref="L2:L201";type="containsText";text="PASS";fill="C6EFCE";"font.color"="006100"}})
[void]$ops.Add(@{command="add";parent="/01_Nhap_cong_thuc";type="conditionalformatting";props=@{ref="L2:L201";type="containsText";text="CHECK";fill="FFC7CE";"font.color"="9C0006"}})
[void]$ops.Add(@{command="add";parent="/06_QC_tom_tat";type="conditionalformatting";props=@{ref="B11";type="containsText";text="SẴN SÀNG";fill="C6EFCE";"font.color"="006100"}})
[void]$ops.Add(@{command="add";parent="/06_QC_tom_tat";type="conditionalformatting";props=@{ref="B11";type="containsText";text="CẦN BỔ SUNG";fill="FFC7CE";"font.color"="9C0006"}})
Batch $ops

officecli save $File | Out-Null
officecli close $File | Out-Null
"Built: $File"
