import tensorflow as tf

a = tf.constant(4)
b = tf.constant(5)

c = tf.add(a, b)

session = tf.Session()

result = session.run(c)

print('adds two constants')
print('--------------------------')  
print(result)

session.close()