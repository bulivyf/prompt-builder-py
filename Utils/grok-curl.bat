@echo off
setlocal

REM Write JSON to a temporary file
set "jsonFile=%TEMP%\grok_payload.json"

(
echo {
echo   "messages": [
echo     {
echo       "role": "system",
echo       "content": "You are a test assistant."
echo     },
echo     {
echo       "role": "user",
echo       "content": "Testing. Just say hi and hello world and nothing else."
echo     }
echo   ],
echo   "model": "grok-4-latest",
echo   "stream": false,
echo   "temperature": 0
echo }
) > "%jsonFile%"

REM Call Grok API
curl -X POST "https://api.x.ai/v1/chat/completions" ^
  -H "Content-Type: application/json" ^
  -H "Authorization: Bearer xai-x5f4TV23E0F9MtyHD5Cg6G8ktJz93a8YWQV5i3cIw81x18sxcS3FgzOoyWZQh2p3IHAvUBPjoIc7SXBm" ^
  --data "@%jsonFile%"

echo.
pause

