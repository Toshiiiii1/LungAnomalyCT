import { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Home from "./components/Home";
import Blogs from "./components/Blog";
import Contact from "./components/Contact";
import NoPage from "./components/NoPage";
import Blogs2 from "./components/Blog2";
import Blogs3 from "./components/Blog3";

function App() {
	return (
		<>
			<BrowserRouter>
				<Routes>
					<Route path="/" element={<Layout />}>
						<Route index element={<Home />} />
						<Route path="blogs" element={<Blogs />} />
						<Route path="blogs2" element={<Blogs2 />} />
						<Route path="blogs3" element={<Blogs3 />} />
						<Route path="contact" element={<Contact />} />
						<Route path="*" element={<NoPage />} />
					</Route>
				</Routes>
			</BrowserRouter>
		</>
	);
}

export default App;
