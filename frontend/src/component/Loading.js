import React from 'react';

const Loading = () => {
  return (
    <div className="h-full flex justify-center items-center">
      <div className="relative w-28 h-28">
        <div className="absolute inset-0 rounded-full border-t-4 border-blue-500 animate-spin"></div>
        <div className="absolute inset-0 flex justify-center items-center">
          <div className="text font-semibold text-blue-500">Loading</div>
        </div>
      </div>
    </div>
  );
};

export default Loading;
