from CNN_Network import network
from utils import plot_confusion_matrix, plot_listen_examples, plot_nn_history

def create_seed():
    return random.randint(1, 1e6)

def classifier(method,
              path_audio,
              path_anntations,
              path_results
              ):
    return None    
    
def Basic_NN(df, 
             X_img,
             path_results,
             epochs = 10, #50
             batch_size = 8, #use 128
             plot_results = False
            ):
    
    
    print ('X_train:',X_train.shape)
    print ('Y_train:',Y_train.shape)
    print ()
    print ('X_val:',X_val.shape)
    print ('Y_val:',Y_val.shape)

    # Call backs to save weights
    filepath= path_results+"/weights_{}.hdf5".format(seed)
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy',verbose=1, save_best_only=True, mode='max')
    #callbacks_list = [checkpoint]

    model = network()
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    print(model.summary())
    
    start = time()

    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), 
              batch_size=batch_size,
              epochs=epochs,
              verbose=2, 
              callbacks=[checkpoint], 
                       )
              #class_weight={0:1.,1:1.})
    end = time()
    
    # ???
    #model.load_weights("Experiments/weights_{}.hdf5".format(seed))
    
    test_loss, test_acc = model.evaluate(X_val, Y_val)
    print(f"Test accuracy: {test_acc:.3f}")

    train_acc = accuracy_score(model.predict_classes(X_train), np.argmax(Y_train,1))
    print('accuracy_score training set:', train_acc)
    val_acc = accuracy_score(model.predict_classes(X_val), np.argmax(Y_val,1))
    print('accuracy_score validation set:', val_acc)
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(np.argmax(Y_val,1), model.predict_classes(X_val))
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    class_names=['0','1']
    print()
    print ('Plotting performance on validation data.')
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

    TN = cnf_matrix[0][0]
    FP = cnf_matrix[0][1]
    FN = cnf_matrix[1][0]
    TP = cnf_matrix[1][1]

    specificity = TN/(TN+FP)
    sensitivity = TP/(FN+TP)

    FPR = 1 - specificity
    FNR = 1 - sensitivity

    performance = []
    performance.append(train_acc)
    performance.append(val_acc)
    performance.append(end-start)

    np.savetxt(path_results+'/train_test_performance_{}.txt'.format(seed), np.asarray(performance), fmt='%f') 

    with open(path_results+'/history_{}.txt'.format(seed), 'wb') as file_out:
        pickle.dump(history.history, file_out)

    #print('Iteration {} ended...'.format(experiment_id))
    print('Results saved to:')
    print(path_results+'/train_test_performance_{}.txt'.format(seed))
    print('-------------------')
    sleep(1)
    
    plot_nn_history(history=history, epochs=epochs)
    plt.show()



