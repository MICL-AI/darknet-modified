with open(r'C:\Users\liang\darknet-modified\weights\tiny.weights', 'rb') as f:
    weight_data = f.read()
    text = weight_data.decode('ascii')
print(text)
# for i in weight_data:
#     d = i.encode()
#     print ("%s" % d)
#    print ("%r, %x, %f, %c, %s" % (i, i, i, i, i))

