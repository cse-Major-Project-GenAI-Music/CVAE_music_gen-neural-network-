import React from 'react';
import '../index.css'
import FilteringImage from '../assets/Pitch_detection.png'
import OriginalAnnotationsImage from '../assets/Original_annotations.png'
import ComparingImage from '../assets/Compare_detected_vs_Original.png'
import SmoothenedImage from '../assets/Predicted_annotations_after_smoothing.png'
import PolyPhonicMusicImage from '../assets/Polyphonic_Midi_music.png'
import PredictionsForCLSTM from '../assets/Predictions_for_CLSTM.png'
import MetricsForCVAE from '../assets/Metrics_for_CVAE.png'
import MetricsForCLSTM from '../assets/Metrics_for_CLSTM.png'

import ActualHumAudio from '../assets/Original_hum.mp3'
import GeneratedMonophonicAudio from '../assets/Generated_instrumental_music.mp3'
import CombinedMusicAudio from '../assets/Mixed_voice_over.mp3'
import PolyphonicMusicAudio from '../assets/Generated_polyphonic_music.wav'
const LandingPage = () => {
  return (
   <div className="h-full flex flex-col" 
         style={{"background": "linear-gradient(to bottom, #3b82f6 10%, #000 30%, #000 70%, #9333ea 100%)"}}
      >
      <h1 className="text-7xl mt-10 mb-20 text-white font-playwriteVN pl-10 leading-[15rem]">
      Generate music from humming
      </h1>

      <AudioPlayerComponent/>
      {/* Grid Section */}
       {/* Grid Section */}
       <div className="p-10">
        {/* First Row */}
        <div className="flex items-center space-x-6 mb-8 justify-between">
          <div className="flex items-center max-w-80">
            <div className="min-w-6 min-h-6 bg-white rounded-full mr-4"> </div>
            <span className="text-white text-2xl">Filtering Detected pitches</span>
          </div>
          <img src={FilteringImage} alt="Filtering Detected Pitches" className="w-3/4 h-auto" />
        </div>

        {/* Second Row */}
        <div className="flex items-center space-x-6 mb-8 justify-between">
          <div className="flex items-center max-w-80">
            <div className="min-w-6 min-h-6 bg-white rounded-full mr-4"></div>
            <span className="text-white text-2xl">Original Annotations</span>
          </div>
          <img src={OriginalAnnotationsImage} alt="Original Annotations" className="w-3/4 h-auto" />
        </div>

        {/* Third Row */}
        <div className="flex items-center space-x-6 mb-8 justify-between">
          <div className="flex items-center  max-w-80">
            <div className="min-w-6 min-h-6 bg-white rounded-full mr-4"></div>
            <span className="text-white text-2xl">Comparing original and detected pitches</span>
          </div>
          <img src={ComparingImage} alt="Comparing Original and Detected Pitches" className="w-3/4 h-auto" />
        </div>

        {/* Fourth Row */}
        <div className="flex items-center space-x-6 mb-8 justify-between">
          <div className="flex items-center  max-w-80">
            <div className="min-w-6 min-h-6 bg-white rounded-full mr-4"></div>
            <span className="text-white text-2xl">Smoothened pitches</span>
          </div>
          <img src={SmoothenedImage} alt="Smoothened Pitches" className="w-3/4 h-auto" />
        </div>

        {/* Fifth Row */}
        <div className="flex items-center space-x-6 mb-8 justify-between">
          <div className="flex items-center  max-w-80">
            <div className="min-w-6 min-h-6 bg-white rounded-full mr-4"></div>
            <span className="text-white text-2xl">Polyphonic music</span>
          </div>
          <div className='w-3/4 h-auto'>
            <img src={PolyPhonicMusicImage} alt="Polyphonic music" className="w-[50%] h-auto" />
          </div>
        </div>

        {/* Sixth Row */}
        <div className="flex items-center space-x-6 mb-8 justify-between">
          <div className="flex items-center  max-w-80">
            <div className="min-w-6 min-h-6 bg-white rounded-full mr-4"></div>
            <span className="text-white text-2xl">CLSTM prediction</span>
          </div>
            <img src={PredictionsForCLSTM} alt="CLSTM prediction" className="w-3/4 h-auto" />
        </div>

        {/* Seventh Row */}
        <div className="flex items-center space-x-6 mb-8 justify-between">
          <div className="flex items-center  max-w-80">
            <div className="min-w-6 min-h-6 bg-white rounded-full mr-4"></div>
            <span className="text-white text-2xl">CVAE metrics</span>
          </div>
            <img src={MetricsForCVAE} alt="CVAE metrics" className="w-3/4 h-auto" />
        </div>

        {/* Eigth Row */}
        <div className="flex items-center space-x-6 mb-8 justify-between">
          <div className="flex items-center  max-w-80">
            <div className="min-w-6 min-h-6 bg-white rounded-full mr-4"></div>
            <span className="text-white text-2xl">CLSTM metrics</span>
          </div>
          <div className='w-3/4 h-auto'>
            <img src={MetricsForCLSTM} alt="CLSTM metrics" className="w-3/4 h-auto" />
          </div>
        </div>

      </div>

   </div>
  );
};

export default LandingPage;


const AudioPlayerComponent = () => {
   return (
      <div className='flex flex-col px-10 mb-20'>
         <div className='text-4xl underline text-white px-10'>
            Illustration
         </div>
         <div className="flex justify-around p-10 ">
            {/* First Section: Actual Hum */}
            <div className="flex flex-col items-center space-y-4 border p-2 rounded-xl border-black">
               <span className="text-white text-xl">Actual Hum</span>
               <audio controls className="w-64">
               <source src={ActualHumAudio} type="audio/mp3" />
               Your browser does not support the audio element.
               </audio>
            </div>
      
            {/* Second Section: Generated Monophonic Music */}
            <div className="flex flex-col items-center space-y-4 border p-2 rounded-xl border-black">
               <span className="text-white text-xl">Generated Monophonic Music</span>
               <audio controls className="w-64">
               <source src={GeneratedMonophonicAudio} type="audio/mp3" />
               Your browser does not support the audio element.
               </audio>
            </div>
      
            {/* Third Section: Combined Music */}
            <div className="flex flex-col items-center space-y-4 border p-2 rounded-xl border-black">
               <span className="text-white text-xl">Combined Music</span>
               <audio controls className="w-64">
               <source src={CombinedMusicAudio} type="audio/mp3" />
               Your browser does not support the audio element.
               </audio>
            </div>
            
            {/* Third Section: Polyphonic Music */}
            <div className="flex flex-col items-center space-y-4 border p-2 rounded-xl border-black">
               <span className="text-white text-xl">Polyphonic Music</span>
               <audio controls className="w-64">
               <source src={PolyphonicMusicAudio} type="audio/mp3" />
               Your browser does not support the audio element.
               </audio>
            </div>
         </div>
      </div>
   );
 };