import random
from pprint import pprint

line = ""
output = ""
'''
with open('range-1-25.txt', 'w') as test_file:
    for x in range(2000):
        line = ""
        for j in range(10):
            line += str(random.randint(1, 25))
            if j != 9:
                line += " "
        #print(line)
        #line.rstrip(line)
        output = output + line + "\n"
    test_file.write(output[:-1])
'''
'''
with open('range-1-25-no-repeats.txt', 'w') as test_file:
    for x in range(2000):
        line = ""
        arr = []
        #for j in range(10):
        #    line += str(random.randint(1, 25))
        #    if j != 9:
        #        line += " "
        while len(arr) != 10:
            num = random.randint(1, 25)
            if num not in arr:
                arr.append(num)
                line += str(num)
                if len(arr) != 10:
                    line += " "
        #print(line)
        #line.rstrip(line)
        output = output + line + "\n"
    test_file.write(output[:-1])
'''

'''
with open('range-minus-5-19.txt', 'w') as test_file:
    for x in range(2000):
        line = ""
        for j in range(10):
            line += str(random.randint(-5, 19))
            if j != 9:
                line += " "
        #print(line)
        #line.rstrip(line)
        output = output + line + "\n"
    test_file.write(output[:-1])
'''

'''
with open('repeated-numbers.txt', 'w') as test_file:
    for x in range(2000):
        line = ""
        for j in range(10):
            line += str(random.randint(1,7))
            if j != 9:
                line += " "
        #print(line)
        #line.rstrip(line)
        output = output + line + "\n"
    test_file.write(output[:-1])
'''

'''
with open('palindrome.txt', 'w') as test_file:
    for x in range(2000):
        line = []

        for j in range(5):
            line.append(str(random.randint(0, 19)))
            if j != 4:
                line += " "
        #print(line)
        #line.rstrip(line)
        #print(line)
        line2=line[::-1]
        l="".join(line)+" "+"".join(line2)

        output = output + l + "\n"
    test_file.write(output[:-1])
'''



with open('data/sort-len-20.txt', 'w') as test_file:
    for x in range(2000):
        line = []
        #for j in range(10):
        #    line.append(random.randint(0, 19))
            #if j != 9:
            #    line += " "
        #line2=line[::-1]
        #l="".join(line)+" "+"".join(line2)
        while len(line) != 20:
            num = random.randint(0, 19)
            if num not in line:
                line.append(num)
        #line = sorted(line)
        #line=line[::-1]
        l=""
        for i in line:
            l += str(i)+" "
        l=l.rstrip()
        output = output + l + "\n"
    test_file.write(output[:-1])

with open('ascending.txt', 'w') as test_file:
    for x in range(2000):
        line = []
        #line = ""
        for j in range(10):
            line.append(random.randint(0, 19))
            #line += str(random.randint(0, 19))
            #if j != 9:
            #    line += " "
        #line2=line[::-1]
        #l="".join(line)+" "+"".join(line2)
        line = sorted(line)
        #line=line[::-1]
        l=""
        for i in line:
            l += str(i)+" "
        l=l.rstrip()
        output = output + l + "\n"
    test_file.write(output[:-1])

'''
with open('data/ascending-train.txt', 'w') as test_file:
    for x in range(10000):
        line = []
        #for j in range(10):
        #    line.append(random.randint(0, 19))
            #if j != 9:
            #    line += " "
        #line2=line[::-1]
        #l="".join(line)+" "+"".join(line2)
        while len(line) != 10:
            num = random.randint(0, 19)
            if num not in line:
                line.append(num)
        line = sorted(line)
        #line=line[::-1]
        l=""
        for i in line:
            l += str(i)+" "
        l=l.rstrip()
        output = output + l + "\n"
    test_file.write(output[:-1])
'''

'''
def read_all_lines(filename):
    with open(filename) as f:
        lines = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    lines = [line.strip() for line in lines]
    return lines

numCountMap = {}
count = 0
lines = read_all_lines("data/range-1-25.txt")
for line in lines:
    numbers = line.split(" ")
    #print(numbers)
    for num in numbers:
        if num in numCountMap:
            numCountMap[num] += 1
            count += 1
        else:
            numCountMap[num] = 1
            count += 1
#print(numCountMap)
pprint(numCountMap)
print(count)
'''