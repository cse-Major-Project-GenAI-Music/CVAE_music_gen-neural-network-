import { createBrowserRouter, RouterProvider, Navigate} from 'react-router-dom';
import HomePage from './component/LandingPage.js';
import DemoPage from './component/DemoPage.js';
import RootLayout from './component/RootLayout.js';

function App() {
  const Router = createBrowserRouter([
    {
      path: '*',
      element: <Navigate to={"/"}/>
    },
    {
      path : "/",
      element :<RootLayout/>,
      children:[
        {
          path : '',
          element : <HomePage/>
        },
        {
          path : 'demo',
          element : <DemoPage/>
        }
      ]
    },
    
  ]);

  return (
    <div className="App">
      {/* provide browser router */}
      <RouterProvider router={Router}/>
    </div>
  );
}


export default App;
