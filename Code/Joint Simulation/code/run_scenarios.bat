for /l %%i in (361, 1, 1000) do (
   echo %%i
   python32 one.py %%i
   ping 127.0.0.1 -n 2 > nul
)