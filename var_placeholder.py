import tensorflow as tf

state = tf.Variable(0)
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

holder = tf.placeholder(tf.float32)
operate_on_holder = 2*holder

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    
    print('increments variable by one')
    print('--------------------------')    
    print(session.run(state))
    for _ in range(5):
        session.run(update)
        print(session.run(state))
    
    print('multiplies placeholder by 2')
    print('--------------------------')    
    print(session.run(operate_on_holder, feed_dict={holder:[4, 5, 6]}))    