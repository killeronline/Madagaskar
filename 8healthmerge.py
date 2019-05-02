dt = []
for i in range(5):
    f = open('meta\health'+str(i)+'.csv','r')        
    dt.append(f.read())
    f.close()

contents = ''.join(dt)
f = open('meta\healthConsolidated.csv','w')
f.write(contents)
f.close()
