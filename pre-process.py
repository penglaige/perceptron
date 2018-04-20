train_source = "data/wsj00-18.pos.txt"
test_source = "data/wsj19-21.pos.txt"
check_source = "data/wsj22-24.pos.txt"

train = "data/train.txt"
test = "data/test.txt"
check = "data/check.txt"

files = [train_source,test_source,check_source]
gets = [train,test,check]
i=0

for file in files:
    with open(file,'r') as f:
        with open(gets[i],'w') as w:
            line = f.readline()
            while(line):
                pairs = line.split()
                for pair in pairs:
                    cut = pair.rfind("/")
                    word = pair[:cut]
                    tag = pair[cut+1:]
                    w.write(word + " " + tag +"\n")
                line = f.readline()
    i += 1
