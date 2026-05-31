$ErrorActionPreference = "Stop"

$Base = if ($env:BASE) { $env:BASE.TrimEnd("/") } else { "http://localhost:8080" }
$UserId = if ($env:DEMO_USER_ID) { $env:DEMO_USER_ID } else { "reviewer-demo-user" }
$Headers = @{}
if ($env:MEMORY_AUTH_TOKEN) {
    $Headers.Authorization = "Bearer $($env:MEMORY_AUTH_TOKEN)"
}

function Wait-ForHealth {
    Write-Host "Waiting for $Base/health ..."
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        try {
            Invoke-RestMethod -Uri "$Base/health" -Method Get | Out-Null
            return
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    throw "Service did not become healthy at $Base"
}

function Send-Turn([string]$SessionId, [string]$Content) {
    $Body = @{
        session_id = $SessionId
        user_id = $UserId
        messages = @(@{ role = "user"; content = $Content })
    } | ConvertTo-Json -Depth 5
    Invoke-RestMethod -Uri "$Base/turns" -Method Post -Headers $Headers `
        -ContentType "application/json" -Body $Body | Out-Null
}

function Assert-Contains([string]$Text, [string]$Expected) {
    if ($Text -notmatch [regex]::Escape($Expected)) {
        throw "Expected '$Expected' was absent"
    }
}

Wait-ForHealth
Write-Host "Cleaning dedicated demo user ..."
Invoke-RestMethod -Uri "$Base/users/$UserId" -Method Delete -Headers $Headers | Out-Null

Send-Turn "demo-1" "I work at Stripe."
Send-Turn "demo-2" "I live in Lisbon."
Send-Turn "demo-3" "My dog is named Biscuit."
Send-Turn "demo-4" "I just joined Notion."

$Memories = Invoke-RestMethod -Uri "$Base/users/$UserId/memories" -Method Get -Headers $Headers
$RecallBody = @{
    query = "What city does the user with the dog named Biscuit live in, and where do they work now?"
    session_id = "demo-probe"
    user_id = $UserId
    max_tokens = 512
} | ConvertTo-Json
$Recall = Invoke-RestMethod -Uri "$Base/recall" -Method Post -Headers $Headers `
    -ContentType "application/json" -Body $RecallBody

$MemoriesJson = $Memories | ConvertTo-Json -Depth 8
$RecallJson = $Recall | ConvertTo-Json -Depth 8
Write-Host ""
Write-Host "Stored memories:"
Write-Host $MemoriesJson
Write-Host ""
Write-Host "Multi-hop recall:"
Write-Host $RecallJson

foreach ($Expected in @("Stripe", "Lisbon", "Biscuit", "Notion")) {
    Assert-Contains $MemoriesJson $Expected
}
foreach ($Expected in @("Lisbon", "Biscuit", "Notion")) {
    Assert-Contains $RecallJson $Expected
}

Write-Host ""
Write-Host "DEMO PASSED"
