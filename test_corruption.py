from src.helpers.images import _check_corrupted_img
import sys

result = _check_corrupted_img('truncated_test.jpg')
print(f'Result: {result}')
if result:
    print('CORRUPTED DETECTED ✅')
else:
    print('CORRUPTED NOT DETECTED ❌')
