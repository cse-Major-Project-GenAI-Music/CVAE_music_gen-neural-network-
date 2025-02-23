import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from ClassCVAE import CVAE
from ClassCLSTM import CLSTM_Decoder

def loadModel_CVAE():
   # Define the device for loading the model.
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print("using device:", device)

   cvae_path = r"D:\acadamics\codes1010\Major_MUSIC\codes\fullstack_me\backend\machine_learning_models\cvae_full_checkpoint.pth"

   # Load the checkpoint. Use map_location if needed.
   checkpoint = torch.load(cvae_path, weights_only=False, map_location=device)

   # Retrieve saved hyperparameters.
   latent_dim = checkpoint['latent_dim']

   # Re-initialize the model with the saved hyperparameters.
   model = CVAE(latent_dim=latent_dim)
   model.load_state_dict(checkpoint['model_state_dict'])
   model.to(device)
   model.eval()  # Set to evaluation mode.

   print("CVAE Model loaded successfully for inference!")

   return model

def loadModel_CLSTM():   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the checkpoint path.
    clstm_path = r"D:\acadamics\codes1010\Major_MUSIC\codes\fullstack_me\backend\machine_learning_models\clstm_decoder_full_checkpoint.pth"

    # Load the checkpoint dictionary.
    checkpoint = torch.load(clstm_path, weights_only=False, map_location=device)  # adjust map_location as needed

    # Extract hyperparameters from the checkpoint.
    latent_dim = checkpoint['latent_dim']
    condition_dim = checkpoint['condition_dim']

    # Re-instantiate the model.
    model = CLSTM_Decoder(latent_dim=latent_dim, condition_dim=condition_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print("CLSTM Model loaded successfully!")
    return model


class Wrapper():
    def __init__(self):
        # Mapping from condition tuple to label name.
        self.model_CVAE  = loadModel_CVAE()
        self.model_CLSTM = loadModel_CLSTM()
        self.name_to_label = {
            (1, 1, 0, 0):  "piano",
            (1, 0, 1, 0):  "guitar",
            (1, 1, 1, 0):  "piano-guitar",
            (1, 1, 0, 1):  "piano-bass",
            (1, 0, 1, 1):  "guitar-bass",
            (1, 1, 1, 1):  "piano-guitar-bass"
        }
    
    def shapeIntoChannels(self, sample):
        """
        Converts the input sample (with discrete values 0,1,2,3) to one-hot channels 
        and computes the condition vector indicating which classes are present.
        """
        sample = sample.long()  # Ensure sample is in long format.
        channel_sample = F.one_hot(sample, num_classes=4).float().permute(2, 0, 1)
        
        # Compute the condition vector: a binary vector of size 4.
        condition = torch.zeros(4, dtype=torch.float)
        unique_labels = torch.unique(sample)
        for label in unique_labels:
            condition[label] = 1.0
    
        return channel_sample, condition

    @staticmethod
    def slerp(t, v0, v1):
        """
        Performs spherical linear interpolation (SLERP) between two latent vectors.
        
        Args:
            t (float): Interpolation value between 0 and 1.
            v0 (torch.Tensor): Starting latent vector of shape (latent_dim,).
            v1 (torch.Tensor): Ending latent vector of shape (latent_dim,).
        
        Returns:
            torch.Tensor: Interpolated latent vector of shape (latent_dim,).
        """
        # Normalize vectors to ensure they lie on the unit hypersphere.
        v0_norm = v0 / torch.norm(v0)
        v1_norm = v1 / torch.norm(v1)
        dot = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)
        
        # If the angle is very small, fall back to linear interpolation.
        if sin_omega < 1e-6:
            return (1 - t) * v0 + t * v1
        
        return (torch.sin((1 - t) * omega) / sin_omega) * v0 + (torch.sin(t * omega) / sin_omega) * v1
    
    def interpolate(self, original, num_samples_required=1, decoder_condition=(1, 1, 1, 1)):
        """
        Encodes the original sample, generates random latent vectors, and uses SLERP
        to combine the original latent vector with each random latent (using t=0.6).
        The resulting latent vectors are then decoded under the specified condition.
        
        Args:
            original (torch.Tensor): A tensor of shape [300, 128] with discrete values.
            num_samples_required (int): Number of new samples to generate.
            decoder_condition (tuple): Condition tuple (e.g., (1,1,1,1)) used by the decoder.
        """
        try:
            if not torch.is_tensor(original):
                return {
                    "message": "received input is not a tensor",
                    "success":  False
                }
            if original.shape != torch.Size([300, 128]):
                return {
                    "message": "received tensor array is not in shape [300, 128]",
                    "success":  False
                }

            # Retrieve label name for the decoder condition.
            label_name = self.name_to_label.get(decoder_condition, "unknown")
            print(f"\n\nGenerating samples for condition {decoder_condition} ({label_name})")
    
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
            print(f'Replicating conditions on {device}...')
            # Create a condition tensor for the decoder: shape (num_samples_required, 4)
            decoder_cond_tensor = torch.tensor(decoder_condition, dtype=torch.float).unsqueeze(0) \
                                   .repeat(num_samples_required, 1).to(device)
            
            print('Generating random latent vectors...')
            latent_dim = self.model_CVAE.latent_dim  # Assumes model is defined globally.
            random_latents = torch.randn(num_samples_required, latent_dim).to(device)
            
            print('Processing original sample...')
            channel_sample, original_condition = self.shapeIntoChannels(original)
            channel_sample = channel_sample.unsqueeze(0).to(device)  # (1, 4, 300, 128)
            original_condition = original_condition.unsqueeze(0).to(device)  # (1, 4)
            
            print('Encoding original sample...')
            mu, logvar = self.model_CVAE.encode(channel_sample, original_condition)
            original_latent = self.model_CVAE.reparameterize(mu, logvar)  # Shape: (1, latent_dim)
            # print(f'Shape of original latent: {original_latent.shape}')
            # print(f'Shape of random latent vectors: {random_latents.shape}')
    
            print('Interpolating original latent with random latents using SLERP...')
            interpolated_latents = []
            for i in range(num_samples_required):
                # Compute interpolated latent using SLERP with t=0.6 (40% original, 60% random).
                interp_latent = Wrapper.slerp(0.6, original_latent.squeeze(0), random_latents[i])
                interpolated_latents.append(interp_latent.unsqueeze(0))
            interpolated_latents = torch.cat(interpolated_latents, dim=0)
    
            print('Decoding interpolated latents of first 30 seconds...')
            decoded_logits = self.model_CVAE.decode(interpolated_latents, decoder_cond_tensor)
            # Convert logits to discrete labels.
            reconstructed_samples = torch.argmax(decoded_logits, dim=1)  # Shape: (num_samples_required, 300, 128)
            
            print('Predicting next 20 seconds of interpolated latents...')
            predicted_logits = self.model_CLSTM(interpolated_latents, decoder_cond_tensor)
            # Convert logits to discrete labels.
            predicted_samples = torch.argmax(predicted_logits, dim=1)  # Shape: (num_samples_required, 200, 128)
        
            generated_samples = torch.cat((reconstructed_samples, predicted_samples), dim=1)  # Shape: (num_samples_required, 500, 128)

            return {
                "message": "interpolation completed",
                "success": True,
                "generated_samples": generated_samples
            }
        
        except Exception as e:
            return {
                "message": f"unexpected error: {e}",
                "success":  False
            }
    

    def overlay_original_with_new_gen_samples(self, original, generated_samples):
        """
        Expands original to (500, 128) and overlays it on each instance of generated_samples.

        Args:
            original (np.ndarray): Array of shape (300, 128) containing the original sample.
            generated_samples (np.ndarray): Array of shape (N, 500, 128) containing N generated samples.

        Returns:
            np.ndarray: Modified array of shape (N, 500, 128) where original overwrites nonzero positions.
        """
        # Ensure inputs are numpy arrays
        original = np.array(original)  # Shape: (300, 128)
        generated_samples = np.array(generated_samples)  # Shape: (N, 500, 128)

        # Expand original to (500, 128) by appending (200, 128) zeros
        padding = np.zeros((200, 128))  # Shape: (200, 128)
        original_expanded = np.vstack((original, padding))  # Shape: (500, 128)

        # Overlay original on every instance of generated_samples
        mixed_samples = np.where(original_expanded != 0, original_expanded, generated_samples)

        return mixed_samples


    def draw_midi_array(self, midi_array, TITLE="", x_len=4, y_len=3):  
        name_of_instrument = {0: "none", 1: "piano", 2: "guitar", 3: "bass"}
        total_timesteps, num_notes = midi_array.shape
        # Exclude 0 values when collecting unique entries.
        unique_values = np.unique(midi_array[midi_array != 0])
        cmap = plt.get_cmap('tab10')
        colors = {val: cmap(i % cmap.N) for i, val in enumerate(unique_values)}
        
        plt.figure(figsize=(x_len, y_len))
        for val in unique_values:
            mask = midi_array == val
            x_coords, y_coords = np.where(mask)
            plt.scatter(x_coords, y_coords, color=colors[val], label=name_of_instrument.get(val, f"Unknown ({val})"), s=1)
        
        plt.xlabel("Timesteps (ds)")
        plt.ylabel("MIDI Note Number")
        plt.title(f"{TITLE}: MIDI Note Activity Over Time")
        plt.legend(markerscale=5, loc='upper right', fontsize='small', ncol=2)
        plt.show()

    def plot_samples(self, samples, x_len=4, y_len=3):
        for i in range(len(samples)):
            self.draw_midi_array(samples[i], TITLE=f"music.{i+1} ", x_len=x_len, y_len=y_len)
    
    def convert_smoothened_freq_to_matrix(self, data, class_label=1):
      def frequency_to_midi(frequency):
         """Convert frequency (Hz) to the nearest MIDI note."""
         if np.isnan(frequency):
            return -1  # Mark as invalid note
         return int(round(69 + 12 * np.log2(frequency / 440.0)))
    
      """Converts input object to a 30000x128 tensor based on MIDI notes."""
      max_timesteps = 3000  # Fixed timesteps representing 30 seconds
      num_notes = 128  # Standard MIDI note range
      
      # Initialize tensor with zeros
      tensor = torch.zeros((max_timesteps, num_notes), dtype=torch.int8)
      
      # Extract time and frequency arrays
      time_array = data['time']
      frequency_array = data['filtered_frequency']
      
      # Convert time to milliseconds (indexing purposes)
      time_ms = (time_array * 100).astype(int)  # Convert seconds to cs index
      
      # Clip to max 30000 timesteps
      valid_indices = time_ms < max_timesteps
      time_ms = time_ms[valid_indices]
      frequency_array = frequency_array[valid_indices]
      
      # Process each time step
      for t, freq in zip(time_ms, frequency_array):
         midi_note = frequency_to_midi(freq)
         if 0 <= midi_note < num_notes:
               tensor[t, midi_note] = class_label  # Mark note as active
         # If midi_note is -1, it remains all zero
      
      return tensor[::10] # returns shape of [300, 128]

