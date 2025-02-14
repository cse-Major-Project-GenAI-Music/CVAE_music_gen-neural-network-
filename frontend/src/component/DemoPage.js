import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import Loading from './Loading'

import Piano_photo from '../assets/Piano_photo.jpeg'
import Guitar_photo from '../assets/Guitar_photo.jpeg'
import Flute_photo from '../assets/Flute_photo_.JPG'
import Synth_photo from '../assets/Synth_photo_.JPG'


const DemoPage = () => {

  const [active, setActive] = useState({ name: "piano", value: "Acoustic_Grand_Piano" });
  const [filePaths, setFilePaths] = useState({});

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  const [recording, setRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);

  const [requestSent, setRequestSent] = useState(JSON.parse(localStorage.getItem("requestSent")) || false);
  const [timer, setTimer] = useState(0);

  const mediaRecorderRef = useRef(null);
  const timerRef = useRef(null);

  
  const options = [
    { name: "piano", value: "Acoustic_Grand_Piano" },
    { name: "flute", value: "Flute" },
    { name: "guitar", value: "Acoustic_Guitar" },
    { name: "synth", value: "SynthStrings_1" },
  ];


  const handleInstrumentChange = (event) => {
    const selectedOption = options.find((option) => option.value === event.target.value);
    if (selectedOption) {
      setActive(selectedOption);
    }
  };


  // Start recording
  const startRecording = async () => {
    setRecording(true);
    setTimer(0);
    
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new MediaRecorder(stream);

    const audioChunks = [];
    mediaRecorderRef.current.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };

    mediaRecorderRef.current.onstop = () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
      const audioUrl = URL.createObjectURL(audioBlob);
      setAudioUrl(audioUrl);
      setAudioBlob(audioBlob);

      // Save audio to local storage
      const reader = new FileReader();
      reader.onloadend = () => {
        localStorage.setItem("audioUrl", reader.result);
      };
      reader.readAsDataURL(audioBlob);
    };

    mediaRecorderRef.current.start();

    // Start the timer
    timerRef.current = setInterval(() => {
      setTimer((prevTimer) => prevTimer + 1);
    }, 1000);
  };

  // Stop recording
  const stopRecording = () => {
    setRecording(false);
    clearInterval(timerRef.current);
    mediaRecorderRef.current.stop();
  };

  // Handle "Record Again"
  const recordAgain = () => {
    setAudioUrl(null);
    setAudioBlob(null);
    setRequestSent(false);
    setTimer(0);

    // Clear local storage
    localStorage.removeItem("audioUrl");
  };

  // Handle "Generate Music"
  const generateMusic = async() => {
    setRequestSent(true);
    setLoading(true);
    localStorage.setItem("requestSent", true);

    if (!audioBlob) {
      alert('No audio recorded to generate music!');
      return;
    }
  
    try {
      const formData = new FormData();
      // Send the raw audio blob directly
      formData.append('audio', audioBlob, 'audio.wav'); // 'audio' is the key, 'audio.wav' is the file name

      const response = await axios.post('http://localhost:5000/generate-music', formData, {
        headers: {
          'Content-Type': 'multipart/form-data', // This tells the server that the body is a form with files
        },
      });
  
      // Handle the response (e.g., displaying the generated audio or processing further)
      console.log('Response:', response.data);
      if (response.data) {
        try {
          // Iterate over the keys in the response data
          Object.entries(response.data).map(([instrumentName, paths]) => {
            // const { instrument, combined } = paths;
            setFilePaths((prevState)=>{
              prevState[instrumentName] = paths;
              return prevState
            })
            console.log(instrumentName, paths);
            return 1;
          });
      
          setError(false);
        } catch (err) {
          console.error('Error generating music:', err);
          alert('Failed to generate music. Please try again later.');
          setError(true);
        }
      }
      
      // Assume the response contains URLs to the generated music
      // setInstrumentalMusic(response.data.instrumentalMusicUrl || null);
      // setCombinedMusic(response.data.combinedMusicUrl || null);
  
      setLoading(false);
    } catch (err) {
      console.error('Error generating music:', err);
      alert('Failed to generate music. Please try again later.');
      setLoading(false);
      setError(true);
    }

  };


  useEffect(() => {
    const storedAudioUrl = localStorage.getItem("audioUrl");
    if (storedAudioUrl) {
      setAudioUrl(storedAudioUrl);
  
      // Fetch the data URL and convert it back to a Blob
      fetch(storedAudioUrl)
        .then(res => res.blob())
        .then(blob => {
          setAudioBlob(blob);
        })
        .catch(err => console.error("Failed to reconstruct audio blob:", err));
    }
  }, []);


  return (
    <div className="flex flex-col items-center gap-4 h-full bg-white px-5 py-5">
      <h1 className="text-4xl underline text-black uppercase">
        Record Voice
      </h1>

      {/* Timer and Recording Animation */}
      {recording ? (
        <div className="flex flex-col items-center mb-5">
          <div className="text-3xl font-bold text-black">{timer}s</div>
          <div className="w-20 h-20 rounded-full border-4 border-blue-500 animate-pulse mt-3"></div>
        </div>
      ) : (
        <div>
          <p className="text text-black">
            {audioUrl ? "Recording Stopped" : "Ready to record"}
          </p>
        </div>
      )}

      {/* Preview of recorded audio */}
      {audioUrl && (
        <div className="mb-1">
          <h2 className="text-lg text-black">Preview:</h2>
          <audio controls className="w-[500px]">
            <source src={audioUrl} type="audio/wav" />
            Your browser does not support the audio element.
          </audio>
        </div>
      )}

      {/* Buttons */}
      {!recording && !audioUrl && (
        <button
          onClick={startRecording}
          className="bg-blue-500 text-white py-2 px-4 rounded"
        >
          Start Recording
        </button>
      )}
      {recording && (
        <button
          onClick={stopRecording}
          className="bg-red-500 text-white py-2 px-4 rounded"
        >
          Stop Recording
        </button>
      )}

      {audioUrl && !loading && (
        <div>
          <button
            onClick={recordAgain}
            className="bg-gray-500 text-white py-2 px-4 rounded mr-4"
          >
            Record Again
          </button>
          <button
            onClick={generateMusic}
            className="bg-green-500 text-white py-2 px-4 rounded"
          >
            Generate Music
          </button>
        </div>
      )}

      <div className="flex justify-center py-2 px-1 gap-3 items-center">
        
          <label htmlFor="instrument-selector" className="mb-2 text-lg font-medium text-gray-700">
            Select an Instrument:
          </label>
          <select
            id="instrument-selector"
            value={active.value}
            onChange={handleInstrumentChange}
            className="p-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            {options.map((option) => (
              <option key={option.value} value={option.value}>
                {option.name}
              </option>
            ))}
          </select>

      </div>

      {
        audioUrl && loading && <div className="w-40 h-40 flex flex-col justify-center">
          <Loading/>
        </div>
      }

      {/* If audio is recorded and generate music is clicked */}
      {requestSent && !loading && !error && (
        <>
          {
            <div className="flex gap-4">
              <div className="flex flex-col items-center gap-4 h-full bg-white px-5 py-5">
        
              {/* Load Instrument Box based on active Instrument */}
              {
                options.map((option, key) =>(
                    <>
                      {
                        option["name"] === active.name &&
                        <div key={key}>
                            {filePaths[active?.value] && (
                                <InstrumentBox instrument={active.name} filePaths={filePaths[active.value]} />
                              )}
                        </div>
                      }
                    </>
                    )
                )
              }
              </div>
              <div>
                {
                  filePaths["polyphonic_music"] && <JukeBox filePaths={filePaths["polyphonic_music"]}/>
                }
              </div>
            </div>
          }
        </>
      )}

      {requestSent && !loading && error && <div className="text-red-500 text-xs">error occured while proccesing </div>}
    </div>
  );
};

export default DemoPage;












const InstrumentBox = ({ instrument, filePaths }) => {
  const [instrumentalMusicUrl, setInstrumentalMusicUrl] = useState(null);
  const [combinedMusicUrl, setCombinedMusicUrl] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const photo = {
    "piano": Piano_photo,
    "flute": Flute_photo,
    "guitar": Guitar_photo,
    "synth": Synth_photo,
  }

  useEffect(() => {
    setLoading(true)
    const fetchFiles = async () => {
      try {
        const response1 = await axios.get(
          `http://localhost:5000/get-file?file_location=${filePaths.instrument}`,
          { responseType: "blob" }
        );
        const blob1URL = URL.createObjectURL(response1.data);

        const response2 = await axios.get(
          `http://localhost:5000/get-file?file_location=${filePaths.combined}`,
          { responseType: "blob" }
        );
        const blob2URL = URL.createObjectURL(response2.data);

        setInstrumentalMusicUrl(blob1URL);
        setCombinedMusicUrl(blob2URL);
        console.log("loaded new instrument music file", filePaths)
        setLoading(false);
      } catch (err) {
        console.error("Error fetching music files:", err);
        setError(true);
        setLoading(false);
      }
    };

    fetchFiles();
  }, [filePaths]);

  if(loading) return <div className="min-w-[500px] border-2 rounded-xl"><Loading/></div>
  return (
    <div className="min-w-[500px] border-2 rounded-xl">
      <div className="w-full h-20">
        <img src={photo[instrument]} alt="image_photo" className="w-full h-full object-cover rounded-t-xl"/>
      </div>
      <h2 className="text-xl text-black mb-4 px-2">Instrument Music ({instrument})</h2>
      {loading ? (
        "Loading..."
      ) : error ? (
        "Error processing"
      ) : (
        <audio controls className="w-[500px] px-2 mb-2">
          <source src={instrumentalMusicUrl} type="audio/wav" />
        </audio>
      )}
      <h2 className="text-xl text-black mb-4 px-2">Combined Music</h2>
      {loading ? (
        "Loading..."
      ) : error ? (
        "Error processing"
      ) : (
        <audio controls className="w-[500px] px-2 mb-2">
          <source src={combinedMusicUrl} type="audio/wav" />
        </audio>
      )}
    </div>
  );
};


const JukeBox = ({filePaths}) => {
  const [audioUrls, setAudioUrls] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    setLoading(true);
    setError(false);
    
    const fetchFiles = async () => {
      try {
        const fetchedUrls = await Promise.all(
          filePaths.map(async (path) => {
            const response = await axios.get(
              `http://localhost:5000/get-file?file_location=${path}`,
              { responseType: "blob" }
            );
            return URL.createObjectURL(response.data);
          })
        );
        setAudioUrls(fetchedUrls);
      } catch (err) {
        console.error("Error fetching music files:", err);
        setError(true);
      } finally {
        setLoading(false);
      }
    };

    fetchFiles();
  }, [filePaths]);

  if (loading) return <div className="min-w-[500px] border-2 rounded-xl">Loading...</div>;
  if (error) return <div className="min-w-[500px] border-2 rounded-xl">Error loading files.</div>;

  return (
    <div className="min-w-[500px] border-2 p-2 rounded-xl">
      <h2 className="text-xl text-black mb-4 px-2">Polyphonic Music</h2>
      {audioUrls.length > 0 ? (
        audioUrls.map((url, index) => (
          <div key={index} className="mb-2 px-2">
            <h3 className="text-lg">Sample {index + 1}</h3>
            <audio controls className="w-[500px]">
              <source src={url} type="audio/wav" />
            </audio>
          </div>
        ))
      ) : (
        <p className="px-2">No audio files available</p>
      )}
    </div>
  );
};