# Documentation for Frontend

## **Frontend Component Documentation**

### **1. `index.js`**
- **Purpose**: This is the entry point of the application that renders the `App` component into the DOM.
- **Functionality**:
  - Imports necessary dependencies like `React`, `ReactDOM`, and `App.js`.
  - Attaches the `App` component to the `root` div in the HTML file.
- **Input**: None.
- **Output**: Renders the entire React application into the root element.

---

### **2. `App.js`**
- **Purpose**: Manages the routing for the application.
- **Functionality**:
  - Uses `react-router-dom` to define the application's routes.
  - Redirects unknown routes to the homepage (`/`).
  - Defines two main routes:
    - `/` (renders `LandingPage`).
    - `/demo` (renders `DemoPage`).
  - Wraps the application within `RootLayout`.
- **Input**: None directly, but handles URL navigation.
- **Output**: The corresponding page (`LandingPage` or `DemoPage`) is rendered based on the current route.

---

### **3. `RootLayout.js`**
- **Purpose**: Provides a consistent layout for the application, including a navigation bar (`Navbar`) and a footer (`Footer`).
- **Functionality**:
  - Renders the `Navbar` at the top and `Footer` at the bottom.
  - Uses `Outlet` from `react-router-dom` to render child components dynamically.
- **Sub-components**:
  - `Navbar`:
    - Displays a navigation button (`Try it out` or `Home page`) based on the current path.
    - Clicking it navigates between `/` and `/demo`.
  - `Footer`:
    - Displays a copyright text.
- **Input**: The current URL path.
- **Output**: Renders the header, footer, and appropriate page content.
- **URL Present At**: `/*` (Handles all routes and renders the correct child components)

---

### **4. `LandingPage.js`**
- **Purpose**: Serves as the homepage, providing an overview of how the application works.
- **Functionality**:
  - Displays a title: "Generate music from humming".
  - Provides an `AudioPlayerComponent` for playing different audio samples:
    - Actual Hum
    - Generated Monophonic Music
    - Combined Music
    - Polyphonic Music
  - Displays a step-by-step illustration of how the humming-to-music transformation works with images.
- **Input**: None.
- **Output**: Informational content with an interactive audio player.
- **URL Present At**: `/`

---

### **5. `DemoPage.js`**
- **Purpose**: Allows users to record their voice, select an instrument, and generate music from the recorded hum.
- **Functionality**:
  - **State Management**:
    - Handles recording state, audio blob, playback URL, request state, and loading/error states.
  - **Recording Audio**:
    - Uses `navigator.mediaDevices.getUserMedia` to capture audio.
    - Stores the recorded audio as a blob.
    - Saves the audio locally for later playback.
  - **Instrument Selection**:
    - Allows users to select from `piano`, `flute`, `guitar`, and `synth`.
  - **Generate Music**:
    - Sends the recorded audio to the backend (`/generate-music` API).
    - Fetches the generated instrumental and combined music files.
  - **Sub-components**:
    - `InstrumentBox`: Displays the generated music for the selected instrument.
    - `JukeBox`: (If included) May handle playback for polyphonic music.
  - **Input**: User’s voice recording, instrument selection.
  - **Output**: Generated instrumental and combined music.
  - **URL Present At**: `/demo`

---

### **6. `Loading.js`**
- **Purpose**: Provides a loading animation while the music generation process is ongoing.
- **Functionality**:
  - Displays a spinning animation with the text "Loading".
- **Input**: None.
- **Output**: Visual feedback for loading state.

---

### **7. `index.css`**
- **Purpose**: Contains global styles for the application.
- **Functionality**:
  - Imports Tailwind CSS utilities.
  - Defines a custom font `Playwrite VN`.
- **Input**: None.
- **Output**: Global styling across the application.

---

### **8. `tailwind.config.js`**
- **Purpose**: Configures Tailwind CSS.
- **Functionality**:
  - Defines custom screen sizes (`mini`, `mobile`, `laptop`).
  - Extends default Tailwind properties.
- **Input**: None.
- **Output**: Provides responsive styling support.

---

### **9. `package.json`**
- **Purpose**: Manages dependencies and scripts for the React application.
- **Functionality**:
  - Defines dependencies:
    - `react`, `react-dom`, `react-router-dom`
    - `axios` (for API calls)
    - `tailwindcss` (for styling)
  - Provides scripts for:
    - `start` - Runs the development server.
    - `build` - Builds the project for production.
    - `test` - Runs test cases.
- **Input**: None.
- **Output**: Dependencies and configurations for the project.

---

