# Backend Documentation

## **Server for Crepe - Documentation**

This server handles audio processing, frequency detection, MIDI conversion, and music generation using **CREPE** and **CVAE** models. It provides multiple endpoints for users to analyze and transform audio files.

---

## **Function Descriptions**

### **1. detect\_frequency\_with\_filter(wav\_file, threshold=0.72, step\_size=10)**

**Purpose:**\
Detects pitch frequencies from a `.wav` file using the **CREPE** model, with a filtering threshold to remove low-confidencea detections.

**Functionality:**

- Loads the audio file.
- Extracts pitch and confidence scores using **CREPE**.
- Filters out low-confidence frequencies based on the given threshold.

**Input:**

- `wav_file` (str): Path to the `.wav` file.
- `threshold` (float, default=0.72): Confidence threshold for pitch detection.
- `step_size` (int, default=10): Determines frame step size for frequency analysis.

**Output:**

- `time` (numpy array): Time stamps of detected frequencies.
- `filtered_frequency` (numpy array): Detected frequencies after filtering.

---

### **2. smoothen\_detected\_frequencies(detected\_frequency)**

**Purpose:**\
Smooths detected frequency data to reduce noise and improve consistency.

**Functionality:**

- Groups detected frequencies to remove abrupt changes.
- Uses time-based segmentation to maintain continuity in frequency transitions.

**Input:**

- `detected_frequency` (dict): Contains time and filtered frequency values.

**Output:**

- `smootheneddetected_frequency` (dict): Contains time and Smoothed frequency values.

---

### **3. construct\_midi\_file\_for\_predicted\_frequencies(smoothened\_detected\_frequency, wav\_file, instrument\_code, instrument\_name)**

**Purpose:**\
Converts detected and smoothed frequencies into a **MIDI file**.

**Functionality:**

- Maps frequencies to MIDI notes.
- Generates a MIDI file with note onset and durations.
- Saves a `.mid` file representing the frequency patterns.

**Input:**

- `smoothened_detected_frequency` (numpy array): Smoothed pitch values.
- `wav_file` (str): Original `.wav` file name.
- `instrument_code` (int): MIDI program number for the instrument.
- `instrument_name` (str): Name of the instrument.

**Output:**

- returns path to generated `.mid` file.
---

### **4. frequency\_to\_midi(frequency)**

**Purpose:**\
Converts a frequency value into a corresponding **MIDI note number**.

**Functionality:**

- Uses logarithmic conversion to map frequency to MIDI scale.

**Input:**

- `frequency` (float): Frequency in Hz.

**Output:**

- `midi_note` (int): Corresponding MIDI note number.

---

### **5. construct\_wav\_file\_from\_midi(input\_midi, instrument\_name, output\_dir)**

**Purpose:**\
Converts a **MIDI file** into a **.wav** file using FluidSynth.

**Functionality:**

- Loads the MIDI file.
- Uses a **soundfont** to synthesize instrument audio.
- Saves the generated `.wav` file.

**Input:**

- `input_midi` (str): Path to the MIDI file.
- `instrument_name` (str): Instrument name for synthesis.
- `output_dir` (str): Directory to save output `.wav` file.

**Output:**

- returns path to generated `.wav` file..

---

### **6. mix\_and\_play\_audio(wav\_file\_1, wav\_file\_2, instrument\_name)**

**Purpose:**\
Mixes two `.wav` files and plays the combined audio.

**Functionality:**

- Loads and aligns two audio tracks.
- Mixes them together parallely.

**Input:**

- `wav_file_1` (str): Path to the first audio file.
- `wav_file_2` (str): Path to the second audio file.
- `instrument_name` (str): Instrument name for context.

**Output:**

- returns path to generated `.wav` file.

---

### **7. Generate\_monophonic\_music(wav\_file)**

**Purpose:**\
Generates a **monophonic** (single-note) music representation from an input `.wav` file, applies different instruments, converts the MIDI file to `.wav`, and returns file paths.

**Functionality:**

- Extracts pitch.
- Converts to MIDI.
- Synthesizes `.wav` files for different instruments.
- Returns paths of generated files and smoothed frequencies.

**Input:**

- `wav_file` (str): Path to the audio file.

**Output:**

- List of paths to generated `.wav` files.
- Smoothed detected frequencies.

---

### **8. Generate\_polyphonic\_music(smoothened\_detected\_frequency, file\_name, num\_of\_required\_samples=2)**

**Purpose:**\
Generates **polyphonic** music (multiple notes at a time), converts frequency data into a matrix, interpolates samples, generates MIDI files, converts them into `.wav`, and returns paths of generated files.

**Functionality:**

- Converts frequencies into a MIDI matrix.
- Interpolates latent space samples.
- Generates a MIDI file.
- Synthesizes `.wav` files.
- Returns paths of generated `.wav` files.

**Input:**

- `smoothened_detected_frequency` (numpy array): Smoothed pitch values.
- `file_name` (str): Name of the output file.
- `num_of_required_samples` (int, default=2): Number of samples to generate.

**Output:**

- List of paths to generated `.wav` files.

---

## **Server Endpoints and Flow**

### **1. /generate-music (POST)**
**Purpose:**  
Processes an input `.wav` file, generates both monophonic and polyphonic music, and returns paths of the newly generated files.

**Flow:**
1. Receives a `.wav` file from the user.
2. Saves the received file at `./inputs/` directory, ensuring a steady **sample rate of 44100 Hz**.
3. Calls `Generate_monophonic_music()` to:
   - Extract frequencies.
   - Generate monophonic MIDI and `.wav` files for different instruments.
   - Return file paths of the generated `.wav` files and smoothed frequencies.
4. Calls `Generate_polyphonic_music()` to:
   - Convert frequency data into a matrix.
   - Interpolate latent space samples.
   - Generate MIDI and `.wav` files.
   - Return file paths of the generated `.wav` files.
5. Returns a dictionary containing paths of all generated files.

**Input:**
- Audio file in `.wav` format.

**Output:**
```json
{
    "monophonic_paths": ["./outputs/mono_instrument1.wav", "./outputs/mono_instrument2.wav"],
    "polyphonic_paths": ["./outputs/poly_sample1.wav", "./outputs/poly_sample2.wav"]
}


### **2. /get_file (GET)**
**Purpose:**  
Retrieves a requested file from the server.

**Flow:**
1. Receives a file name as a query parameter.
2. Checks if the file exists in the output directory.
3. If the file exists, returns the file as a response.
4. If the file does not exist, returns a 404 error.

**Input:**
- File name (query parameter).

**Output:**
- If successful: The requested file.
- If unsuccessful:
```json
{
    "error": "File not found"
}
```

---

# Helper Modules

## **MatrixToMidi.py**
Converts a `[300, 128]` matrix into a MIDI file.

### **1. `generate_midi_from_matrix(sample, output_path)`**
- **Purpose**: Converts a MIDI matrix to a `.mid` file.
- **Functionality**:
  - Maps matrix values to MIDI instruments.
  - Iterates through time steps and notes.
  - Handles note activation and deactivation.
  - Saves the `.mid` file.
- **Input**:
  - `sample (torch.Tensor)`: MIDI representation `[300, 128]`.
  - `output_path (str)`: Path to save the MIDI file.
- **Output**: Saves a `.mid` file.

---

# Machine Learning Utilities

## **ClassCVAE.py**
Defines the Conditional Variational Autoencoder (CVAE) model.

### **1. `__init__(latent_dim, condition_dim)`**
- **Purpose**: Initializes the CVAE model.
- **Functionality**:
  - Defines encoder, decoder, and latent layers.
  - The latent vector is concatenated with condition labels.
- **Input**:
  - `latent_dim (int)`: Dimension of latent space.
  - `condition_dim (int)`: Condition vector dimension.
- **Output**: Initialized CVAE model.

### **2. `encode(x, condition)`**
- **Purpose**: Encodes an input sample into latent space.
- **Functionality**:
  - Processes the input through convolutional layers.
  - Concatenates the condition vector.
  - Computes mean (`mu`) and log variance (`logvar`).
- **Input**:
  - `x (torch.Tensor)`: Input tensor `[batch_size, 4, 300, 128]`.
  - `condition (torch.Tensor)`: Condition vector `[batch_size, 4]`.
- **Output**: `(mu, logvar)`.

### **3. `reparameterize(mu, logvar)`**
- **Purpose**: Samples from the latent space.
- **Functionality**:
  - Applies the reparameterization trick.
- **Input**:
  - `mu (torch.Tensor)`, `logvar (torch.Tensor)`.
- **Output**: Latent vector `z`.

### **4. `decode(z, condition)`**
- **Purpose**: Decodes latent vectors into output.
- **Functionality**:
  - Concatenates `z` with `condition`.
  - Passes through transposed convolutional layers.
- **Input**:
  - `z (torch.Tensor)`: Latent vector `[batch_size, latent_dim]`.
  - `condition (torch.Tensor)`: `[batch_size, 4]`.
- **Output**: Logits `[batch_size, 4, 300, 128]`.

### **5. `forward(x, condition)`**
- **Purpose**: Defines the forward pass.
- **Functionality**:
  - Encodes input.
  - Reparameterizes.
  - Decodes.
- **Input**: `x, condition`
- **Output**: `(logits, mu, logvar)`.



## **WrapperForCVAE.py**
This script provides a wrapper for the Conditional Variational Autoencoder (CVAE), handling model loading, input preprocessing, latent space manipulation, and output generation.

### **1. `loadModel()`**
- **Purpose**: Loads the pre-trained CVAE model from a checkpoint file.
- **Functionality**:
  - Detects the device (`cuda` if available, else `cpu`).
  - Loads model parameters from `cvae_full_checkpoint.pth`.
  - Initializes a CVAE model with the saved hyperparameters.
  - Loads the saved model state dictionary and sets it to evaluation mode.
- **Input**: None
- **Output**: The loaded `CVAE` model.

### **2. `Wrapper` (Class)**
Encapsulates the `CVAE` model and provides methods for encoding, decoding, and interpolating latent space representations.

### **3. `shapeIntoChannels(sample)`**
- **Purpose**: Converts input samples into one-hot encoded channels and generates a condition vector.
- **Functionality**:
  - Converts the sample into long format.
  - Performs one-hot encoding (num_classes = 4).
  - Computes a binary condition vector indicating which labels are present.
- **Input**: `sample (torch.Tensor)` of shape `[300, 128]`, containing values {0,1,2,3}.
- **Output**: `(channel_sample, condition)`:
  - `channel_sample`: One-hot encoded tensor of shape `[4, 300, 128]`.
  - `condition`: Binary vector of shape `[4]`.

### **4. `slerp(t, v0, v1)`**
- **Purpose**: Performs Spherical Linear Interpolation (SLERP) between two latent vectors.
- **Functionality**:
  - Normalizes the vectors.
  - Computes the angle and performs interpolation.
- **Input**:
  - `t (float)`: Interpolation weight.
  - `v0, v1 (torch.Tensor)`: Latent vectors.
- **Output**: Interpolated latent vector.

### **5. `interpolate(original, num_samples_required, decoder_condition)`**
- **Purpose**: Encodes a sample, generates random latent vectors, interpolates using SLERP, and decodes under a specified condition.
- **Functionality**:
  - Validates input tensor shape.
  - Encodes the original sample.
  - Generates random latent vectors.
  - Uses SLERP to interpolate.
  - Decodes interpolated latents to generate new samples.
- **Input**:
  - `original (torch.Tensor)`: Shape `[300, 128]`, input music representation.
  - `num_samples_required (int)`: Number of generated samples.
  - `decoder_condition (tuple)`: Specifies the instrument condition.
- **Output**: Dictionary containing:
  - `"generated_samples"`: Generated MIDI tensor of shape `[num_samples_required, 300, 128]`.

### **6. `overlay_original_with_new_gen_samples(original, generated_samples)`**
- **Purpose**: Merges nonzero values from the original sample into the generated samples.
- **Functionality**:
  - Uses `np.where` to replace zero values in generated samples with corresponding original values.
- **Input**:
  - `original (np.ndarray)`: Shape `[300, 128]`, original sample.
  - `generated_samples (np.ndarray)`: Shape `[N, 300, 128]`, generated samples.
- **Output**: Merged samples of shape `[N, 300, 128]`.

### **7. `draw_midi_array(midi_array, TITLE, x_len, y_len)`**
- **Purpose**: Visualizes a MIDI array as a scatter plot.
- **Input**: 
  - `midi_array (np.ndarray)`: Shape `[300, 128]`, MIDI representation.
  - `TITLE (str)`: Plot title.
- **Output**: A scatter plot.

### **8. `plot_samples(samples, x_len, y_len)`**
- **Purpose**: Plots multiple MIDI samples.
- **Functionality**:
  - Calls `draw_midi_array` for each sample.
- **Input**: 
  - `samples (list of np.ndarray)`: List of MIDI samples.





