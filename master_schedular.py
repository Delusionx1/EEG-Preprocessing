import subprocess
import time
from itertools import permutations

gpu_num = 1
participant_amount = 1
start_participant_num = 1
process_num = 2

all_p = []

f = open("time.txt",'a')
end_data_loading = time.time()

list1 = list(permutations([1, 2, 3, 0], 4))
list2 = list(permutations([1, 2, 3, 0], 3))
list3 = list(permutations([1, 2, 3, 0], 2))
list4 = list(permutations([1, 2, 3, 0], 1))

print(list1,list2,list3,list4)
allnum = len(list1)+len(list2)+len(list3)+len(list4)
allList = list1 + list2 + list3 + list4

print(allnum,len(allList))
strList = ['']
for i,content in enumerate(allList):
  temp = ''
  for j in content:
    temp = str(j) + temp
  strList.append(temp)
print(strList)

all_p = []
orders_amount = len(strList)
current_process_count = 0
total_process_count = orders_amount * ((participant_amount- start_participant_num)+1)
for i in range(process_num):
    p = subprocess.Popen(["python", "main_test.py", "-o", \
	strList[current_process_count % orders_amount], '-p', \
	str(int(current_process_count / orders_amount) + start_participant_num), \
	'-g', str(i % gpu_num), '-m', 'FBCNet', '-d', 'BCIIV2a'])
    all_p.append(p)
    current_process_count += 1

all_return_code = [1 for i in range(process_num)]

while True:
    for i in range(len(all_p)):
        all_return_code[i] = all_p[i].poll()
        if(all_return_code[i] == 0):
            if(current_process_count < total_process_count):
                all_p[i] = subprocess.Popen(["python", "main_test.py", "-o", \
                strList[current_process_count % orders_amount], '-p', \
                str(int(current_process_count / orders_amount) + start_participant_num), \
                '-g', str(i % gpu_num), '-m', 'FBCNet', '-d', 'BCIIV2a'])
                current_process_count += 1

    for i in range(len(all_p)):
        all_return_code[i] = all_p[i].poll()

    if all(i==0 for i in all_return_code):
        break

end = time.time()
f.write("Training time consumed: "+str(end - end_data_loading)+"\n")
f.close()
