from display_loss import *
F = open('list.txt','r')

loss= []
count = 0
for ligne in F.readlines():
    if count % 10 ==0:
        loss += [float(ligne[-7:-2])]
    count += 1

display_loss_norm(loss)
print(loss)
F.close()
