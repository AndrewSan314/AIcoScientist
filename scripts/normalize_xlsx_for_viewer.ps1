$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem

$Source = "F:\AI\GTIP\outputs\Mau_nhap_du_lieu_vat_lieu_cho_AI_v2_truc_quan.xlsx"
$Target = "F:\AI\GTIP\outputs\Mau_nhap_du_lieu_vat_lieu_cho_AI_v2_office_viewer.xlsx"
$MainNs = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"

function Read-EntryText($zip, [string]$name) {
    $entry = $zip.GetEntry($name)
    if (-not $entry) { throw "Missing entry: $name" }
    $reader = New-Object System.IO.StreamReader($entry.Open(), [Text.UTF8Encoding]::new($false))
    try { $reader.ReadToEnd() } finally { $reader.Dispose() }
}

function Write-EntryText($zip, [string]$name, [string]$text) {
    $entry = $zip.CreateEntry($name, [IO.Compression.CompressionLevel]::Optimal)
    $writer = New-Object System.IO.StreamWriter($entry.Open(), [Text.UTF8Encoding]::new($false))
    try { $writer.Write($text) } finally { $writer.Dispose() }
}

function Copy-Entry($srcZip, $dstZip, [string]$sourceName, [string]$targetName) {
    $src = $srcZip.GetEntry($sourceName)
    if (-not $src) { throw "Missing entry: $sourceName" }
    $dst = $dstZip.CreateEntry($targetName, [IO.Compression.CompressionLevel]::Optimal)
    $input = $src.Open(); $output = $dst.Open()
    try { $input.CopyTo($output) } finally { $input.Dispose(); $output.Dispose() }
}

function Normalize-MainNamespace([string]$xml, [bool]$convertPlainStrings) {
    $xml = $xml.Replace('xmlns:x="' + $MainNs + '"', 'xmlns="' + $MainNs + '"')
    $xml = $xml.Replace('<x:', '<').Replace('</x:', '</')
    if (-not $convertPlainStrings) { return $xml }

    $doc = New-Object System.Xml.XmlDocument
    $doc.PreserveWhitespace = $true
    $doc.LoadXml($xml)
    $ns = New-Object System.Xml.XmlNamespaceManager($doc.NameTable)
    $ns.AddNamespace('m', $MainNs)
    $cells = @($doc.SelectNodes('//m:c[@t="str" and not(m:f)]', $ns))
    foreach ($cell in $cells) {
        $valueNode = $cell.SelectSingleNode('m:v', $ns)
        if (-not $valueNode) { continue }
        $value = $valueNode.InnerText
        $inline = $doc.CreateElement('is', $MainNs)
        $text = $doc.CreateElement('t', $MainNs)
        if ($value.Length -ne $value.Trim().Length) {
            $space = $doc.CreateAttribute('xml', 'space', 'http://www.w3.org/XML/1998/namespace')
            $space.Value = 'preserve'; [void]$text.Attributes.Append($space)
        }
        $text.InnerText = $value
        [void]$inline.AppendChild($text)
        [void]$cell.ReplaceChild($inline, $valueNode)
        $cell.SetAttribute('t', 'inlineStr')
    }
    $settings = New-Object System.Xml.XmlWriterSettings
    $settings.Encoding = [Text.UTF8Encoding]::new($false)
    $settings.Indent = $false
    $settings.OmitXmlDeclaration = $false
    $settings.CloseOutput = $true
    $sw = New-Object System.IO.StringWriter
    $xw = [Xml.XmlWriter]::Create($sw, $settings)
    $doc.Save($xw); $xw.Close()
    return $sw.ToString().Replace('encoding="utf-16"', 'encoding="utf-8"')
}

$sheetNames = @(
    '00_Huong_dan',
    '01_Nhap_cong_thuc',
    '02_Ket_qua_SEM_EDX',
    '03_Ket_qua_dien_hoa',
    '04_Phan_hoi_AI',
    '06_QC_tom_tat',
    '06_Mapping_AI'
)
$sourceSheets = @('sheet1.xml','sheet2.xml','sheet3.xml','sheet4.xml','sheet5.xml','sheet7.xml','sheet8.xml')

$workbookSheets = New-Object Text.StringBuilder
for ($i=0; $i -lt $sheetNames.Count; $i++) {
    $id = $i + 1
    $state = if ($i -eq 6) { ' state="hidden"' } else { '' }
    [void]$workbookSheets.Append('<sheet name="' + $sheetNames[$i] + '" sheetId="' + $id + '"' + $state + ' r:id="rId' + $id + '"/>')
}
$workbookXml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' +
    '<workbook xmlns="' + $MainNs + '" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">' +
    '<bookViews><workbookView activeTab="0"/></bookViews><sheets>' + $workbookSheets.ToString() +
    '</sheets><calcPr fullCalcOnLoad="1"/></workbook>'

$rels = New-Object Text.StringBuilder
for ($i=1; $i -le 7; $i++) {
    [void]$rels.Append('<Relationship Id="rId' + $i + '" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet' + $i + '.xml"/>')
}
[void]$rels.Append('<Relationship Id="rId8" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>')
[void]$rels.Append('<Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>')
$workbookRels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + $rels.ToString() + '</Relationships>'

$overrides = New-Object Text.StringBuilder
[void]$overrides.Append('<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>')
[void]$overrides.Append('<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>')
[void]$overrides.Append('<Override PartName="/xl/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>')
for ($i=1; $i -le 7; $i++) { [void]$overrides.Append('<Override PartName="/xl/worksheets/sheet' + $i + '.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>') }
[void]$overrides.Append('<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>')
[void]$overrides.Append('<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>')
$contentTypes = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>' + $overrides.ToString() + '</Types>'

$rootRels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>'

$titleXml = ($sheetNames | ForEach-Object { '<vt:lpstr>' + [Security.SecurityElement]::Escape($_) + '</vt:lpstr>' }) -join ''
$appXml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Microsoft Excel</Application><TitlesOfParts><vt:vector xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes" size="7" baseType="lpstr">' + $titleXml + '</vt:vector></TitlesOfParts></Properties>'

$src = [IO.Compression.ZipFile]::OpenRead($Source)
try {
    if (Test-Path -LiteralPath $Target) { Remove-Item -LiteralPath $Target -Force }
    $stream = [IO.File]::Open($Target, [IO.FileMode]::CreateNew)
    $dst = New-Object IO.Compression.ZipArchive($stream, [IO.Compression.ZipArchiveMode]::Create, $false)
    try {
        Write-EntryText $dst '[Content_Types].xml' $contentTypes
        Write-EntryText $dst '_rels/.rels' $rootRels
        Copy-Entry $src $dst 'docProps/core.xml' 'docProps/core.xml'
        Write-EntryText $dst 'docProps/app.xml' $appXml
        Write-EntryText $dst 'xl/workbook.xml' $workbookXml
        Write-EntryText $dst 'xl/_rels/workbook.xml.rels' $workbookRels
        $styles = Normalize-MainNamespace (Read-EntryText $src 'xl/styles.xml') $false
        Write-EntryText $dst 'xl/styles.xml' $styles
        Copy-Entry $src $dst 'xl/theme/theme1.xml' 'xl/theme/theme1.xml'
        for ($i=0; $i -lt $sourceSheets.Count; $i++) {
            $xml = Normalize-MainNamespace (Read-EntryText $src ('xl/worksheets/' + $sourceSheets[$i])) $true
            Write-EntryText $dst ('xl/worksheets/sheet' + ($i+1) + '.xml') $xml
        }
    } finally { $dst.Dispose(); $stream.Dispose() }
} finally { $src.Dispose() }

Write-Output $Target
