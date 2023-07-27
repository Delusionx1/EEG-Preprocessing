import subprocess
import time

gpu_num = 1
participant_amount = 9
current_participant_num = 1
process_num = 3

all_p = []

for i in range(process_num):
    p = subprocess.Popen(["python", "main_test.py", "-o", \
        '213', '-p', str(i+1), '-g', str(0), '-m', 'FBCNet', '-d', 'BCIIV2a'])    
    all_p.append(p)
    current_participant_num += 1

all_return_code = [1 for i in range(process_num)]

while True:
    for i in range(len(all_p)):
        all_return_code[i] = all_p[i].poll()
        if(all_return_code[i] == 0):
            if(current_participant_num <= participant_amount):
                all_p[i] = subprocess.Popen(["python", "main_test.py", "-o", '213', '-p', \
                    str(current_participant_num), '-g', str(0), '-m', 'FBCNet', '-d', 'BCIIV2a']) 
                current_participant_num += 1

    for i in range(len(all_p)):
        all_return_code[i] = all_p[i].poll()

    if all(i==0 for i in all_return_code):
        break
