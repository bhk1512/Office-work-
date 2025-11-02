import sys
try:
    import dashboard.callbacks as cb
    print('OK')
except Exception as e:
    print('ERR', type(e).__name__, e)
