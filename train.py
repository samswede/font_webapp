

from utils import *
from preprocessing import *
from NEW_variational_autoencoder import *


num_epochs = 700
model_size = 'big'

for epoch in range(num_epochs):

   train_loss = train_epoch(vae, device, train_loader, optim)
   val_loss = test_epoch(vae, device, valid_loader)
   if 9 == epoch % 10:
      print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))
      plot_ae_outputs(vae.encoder,vae.decoder,n=9)
   if 99 == epoch % 100:
      model_save_name = '{}_vae_L{}_E{}.pt'.format(model_size, d, epoch+1)
      model_path = F"/content/drive/MyDrive/FontFinder/Models/VAEs/{model_save_name}" 
      torch.save(vae.state_dict(), model_path)