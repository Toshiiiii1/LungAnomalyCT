import "./App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import UploadMHDRAW from "./components/UploadMHDRAW";
import History from "./components/History";
import NoPage from "./components/NoPage";
import UploadImage from "./components/UploadImage";
import UploadDCM from "./components/UploadDCM";

function App() {
	return (
		<>
			<BrowserRouter>
				<Routes>
					<Route path="/" element={<Layout />}>
						<Route index element={<History />} />
						<Route path="upload_mhdraw" element={<UploadMHDRAW />} />
						<Route path="upload_image" element={<UploadImage />} />
						<Route path="upload_dcm" element={<UploadDCM />} />
						<Route path="*" element={<NoPage />} />
					</Route>
				</Routes>
			</BrowserRouter>
		</>
	);
}

export default App;
