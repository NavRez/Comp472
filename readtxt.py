from numpy import loadtxt
lines = loadtxt("Naive-Bayes-dataset.txt", dtype=str,comments="#", delimiter=",", unpack=False)
# print(lines)

n=0
p=0

for i in range(0,2383):
    if lines[i][1] == 'pos':
        p+=1
    else:
        n+=1

print("# of neg in GEN is ", n)
print("# of pos in GEN is ",p) 

#OG operation
lines2 = loadtxt("verify.txt", dtype=str,comments="#", delimiter="txt", unpack=False)
line3=list()


for i in range(0,2383):
    line3.append(lines2[i][0])
# lines2[i][0]

line4=list()

for i in range(0,2383):
    line4.append(line3[i].split(" "))

# print(line3)


# print(line3[0].split(" "))

# print(line4)

OG=list()

for i in range(0,2383):
    OG.append(line4[i][1])

# print(OG)


# OG count
n1=0
p1=0
for i in range(0,2383):
    if OG[i] == 'pos':
        p1+=1
    else:
        n1+=1

print("OG neg # are", n1)
print("OG pos # are", p1)

#Which line are different

GEN=list()

for i in range(0,2383):
    GEN.append(lines[i][1])

# print(GEN)

Differences=list()

for i in range(0,2383):
    if GEN[i] != OG[i]:
        Differences.append(i+1)

print(Differences)