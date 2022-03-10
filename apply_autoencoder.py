# EPS for FGSM attack
curr_attack = "FGSM_Linfinity"
eps_values = np.array([0.000, 0.001, 0.002, 0.003, 0.005, 0.007, 
                       0.010, 0.020, 0.030])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current Device: ", device)
print(torch.cuda.get_device_name(torch.cuda.current_device()))

filename = "RML2016.10b.dat"
snrs, mods, X, labels = process_data(filename)

x_train, x_test, y_train, y_test, test_labels, test_idx = train_test_split(X, labels, mods)

train_dataset = TensorDataset(x_train, x_train)
test_dataset  = TensorDataset(x_test,  x_test)
batch_size = 16
TrainLoader = DataLoader(train_dataset, batch_size = batch_size, 
                         shuffle = False)
TestLoader  = DataLoader(test_dataset,  batch_size = batch_size, 
                         shuffle = False)

model = CNN(input_size = 128)
model.load_state_dict(torch.load("CNN_base.pt"))
model = model.to(device)
encoder = AUTOENCODER()
encoder.load_state_dict(torch.load("AUTOENCODER.pt"))
encoder.to(device)
criterion  = nn.MSELoss()

T = []
for eps in eps_values:
    E = []
    for batch_idx, (data, lbs) in enumerate(TrainLoader) :
        data = data.to(device)
        lbs = lbs.to(device)

        #FGSM L-infinity Adversarial Retraining
        adv_data = FGSM_Linf_attack(model, device, data, labels, eps)

        outputs = encoder(adv_data)
        loss = criterion(outputs,lbs)
        E.append(loss)
    
    T.append(max(E))
    
    
encoder.eval()
dtct_rate = []
for i in range(len(eps_values)) 
    detect_num = 0
    all_num = 0
    for X_val_batch, y_val_batch in TestLoader :
        X_val_batch = X_val_batch.to(device)
        y_val_batch = y_val_batch.to(device)
        
        adv_x = FGSM_Linf_attack(model, device, x_val_batch, y_val_batch, eps)
        y_val_pred = encoder(adv_x)
        val_loss = criterion(y_val_pred, adv_x)

        if val_loss > T:
            detect_num += 1
        all_num += 1
    
    detection_rate = float(detect_num) / float(all_num)  
    print(f'eps value {eps_values[i]}: Detection rate: {detection_rate:.3f}')   
    dtct_rate.append(detection_rate)

print(dict_rate)
