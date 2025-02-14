import React from 'react'
import { Outlet } from 'react-router-dom';
import { Link, useLocation } from 'react-router-dom';

const RootLayout = () => {
  return (
   <div className="flex flex-col min-h-screen">
      <Navbar />
      <main className="flex-grow flex">
      <div className="flex-grow">
         <Outlet />
      </div>
      </main>
      <Footer />
   </div>
  )
}

export default RootLayout



const Navbar = () => {
   const location = useLocation(); // To get the current URL
 
   return (
     <nav className="w-full relative text-white">
       {/* Right Section (Button-like Link) */}
       <div className="fixed top-4 right-4 text-xl">
         <Link
           to={location.pathname === '/' ? '/demo' : '/'}
           className="inline-block py-2 px-4 bg-green-500 text-white rounded-lg hover:bg-green-600 cursor-pointer"
         >
           {location.pathname === '/' ? 'Try it out' : 'Home page'}
         </Link>
       </div>
     </nav>
   );
 };


function Footer() {
   return (
     <footer className="bg-gray-800 text-white py-4">
       <div className="container mx-auto text-center">
         <p>2025: Music generated from hum</p>
       </div>
     </footer>
   )
 }
