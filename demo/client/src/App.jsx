import { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

function App() {
	const [file1, setFile1] = useState(null);
	const [file2, setFile2] = useState(null);
	const [result, setResult] = useState([]); // Danh sách ảnh từ API
	const [currentIndex, setCurrentIndex] = useState(0); // Ảnh hiện tại
	const [predictIndex, setPredictIndex] = useState(0); // Ảnh hiện tại
	const [predictedImageUrl, setPredictedImageUrl] = useState(""); // URL ảnh đã xử lý
	const [isLoading, setIsLoading] = useState(false);
	const [history, setHistory] = useState([]);
	const [selectedHistory, setSelectedHistory] = useState(null);

	const handleFile1Change = (e) => setFile1(e.target.files[0]);
	const handleFile2Change = (e) => setFile2(e.target.files[0]);

	const handleUpload = async () => {
		if (!file1 || !file2) {
			alert("Vui lòng chọn cả hai tệp!");
			return;
		}

		const formData = new FormData();
		formData.append("file1", file1);
		formData.append("file2", file2);

		setIsLoading(true);
		try {
			const response = await axios.post(
				"http://localhost:8000/upload",
				formData,
				{
					headers: {
						"Content-Type": "multipart/form-data",
					},
				}
			);
			setResult(response.data.files);
			setCurrentIndex(0);
			setPredictedImageUrl("");
		} catch (error) {
			console.error("Upload failed:", error);
			alert("Tải lên thất bại.");
		} finally {
			setIsLoading(false); // Kết thúc tải ảnh
		}
	};

	const fetchHistory = async () => {
		try {
			const response = await axios.get("http://localhost:8000/history");
			setHistory(response.data.history);
		} catch (error) {
			console.error("Failed to fetch history:", error);
		}
	};

	useEffect(() => {
		fetchHistory();
	}, []);

	// Xử lý cuộn chuột để chuyển ảnh
	const handleScroll = (event) => {
		if (event.deltaY > 0) {
			// Cuộn xuống: Hiển thị ảnh kế tiếp
			setCurrentIndex((prevIndex) =>
				Math.min(prevIndex + 1, result.length - 1)
			);
		} else if (event.deltaY < 0) {
			// Cuộn lên: Hiển thị ảnh trước đó
			setCurrentIndex((prevIndex) => Math.max(prevIndex - 1, 0));
		}
	};

	// Khi nhấn vào một thẻ, hiển thị ảnh của lịch sử đó
	const handleHistoryClick = (record) => {
		setSelectedHistory(record);
		setResult(record.extracted_images);
		setCurrentIndex(record.slice_index);
		setPredictIndex(record.slice_index);
		setPredictedImageUrl(record.predicted_image);
	};

	const handleDeleteHistory = async (recordId) => {
		try {
			await axios.delete(`http://localhost:8000/history/${recordId}`);
			fetchHistory();
			setResult([]);
			alert("Xóa lịch sử thành công.");
		} catch (error) {
			console.error("Failed to delete history:", error);
			alert("Xóa lịch sử thất bại.");
		}
	};

	const handlePredict = () => {
		const currentImageUrl = result[currentIndex];

		axios
			.post("http://localhost:8080/predict/", {
				image_url: currentImageUrl,
			})
			.then((response) => {
				setPredictedImageUrl(response.data.predicted_image); // Lưu URL ảnh đã xử lý
				setPredictIndex(currentIndex);
				fetchHistory();
			})
			.catch((error) => {
				console.error("Error predicting image:", error);
			});

		console.log(predictedImageUrl);
	};

	return (
		<>
			<div className="file_uploader">
				<h1>Tải lên 2 file .mhd và .raw</h1>
				<input type="file" accept=".mhd" onChange={handleFile1Change} />
				<input type="file" accept=".raw" onChange={handleFile2Change} />
				<button onClick={handleUpload}>Tải lên</button>
			</div>

			<div className="App">
				{isLoading ? (
					<p>Loading images...</p>
				) : result.length > 0 ? (
					<>
						{/* Khung chứa ảnh hiển thị và ảnh dự đoán */}
						<div
							style={{
								display: "flex",
								justifyContent: "center",
								alignItems: "center",
							}}
						>
							{/* Ảnh hiển thị */}
							<div>
								<img
									src={result[currentIndex]}
									alt={`Slice ${currentIndex}`}
									style={{
										margin: "20px", // Thêm khoảng cách bên trái và bên phải
									}}
									onWheel={handleScroll}
								/>
								<p
									style={{
										marginTop: "20px",
										fontSize: "18px",
									}}
								>
									Slice {currentIndex} of {result.length}
								</p>
							</div>

							{predictedImageUrl && (
								<div style={{ textAlign: "center" }}>
									<img
										src={predictedImageUrl}
										alt="Predicted"
										style={{
											margin: "20px", // Thêm khoảng cách bên trái và bên phải
										}}
									/>
									<p
										style={{
											marginTop: "20px",
											fontSize: "18px",
										}}
									>
										Predicted Image:{" "}
										{`Slice ${predictIndex}`}
									</p>
								</div>
							)}
						</div>

						{/* Các nút điều khiển Previous và Next */}
						<div style={{ marginTop: "20px" }}>
							<button
								onClick={() =>
									setCurrentIndex((prevIndex) =>
										prevIndex > 0
											? prevIndex - 1
											: result.length - 1
									)
								}
								style={{
									marginRight: "10px",
									padding: "10px",
									fontSize: "16px",
									cursor: "pointer",
								}}
							>
								Previous
							</button>
							<button
								onClick={() =>
									setCurrentIndex((prevIndex) =>
										prevIndex < result.length - 1
											? prevIndex + 1
											: 0
									)
								}
								style={{
									marginLeft: "10px",
									padding: "10px",
									fontSize: "16px",
									cursor: "pointer",
								}}
							>
								Next
							</button>
							<button
								onClick={handlePredict}
								style={{
									marginLeft: "10px",
									padding: "10px",
									fontSize: "16px",
									cursor: "pointer",
								}}
							>
								Predict
							</button>
						</div>
					</>
				) : (
					<p>No images to display</p>
				)}
			</div>
			<h2>Lịch sử nhận diện</h2>
			<div className="history-container">
				{history.map((record) => (
					<div
						key={record.id}
						className="history-card"
						onClick={() => handleHistoryClick(record)}
					>
						{/* Hiển thị thông tin file_id */}
						<h3>File ID: {record.file_id}</h3>

						{/* Hiển thị thông tin tên file .mhd và .raw */}
						<p>File MHD: {record.file1_name}</p>
						<p>File RAW: {record.file2_name}</p>

						{/* Hiển thị thông tin chỉ số lát cắt (slice_index) */}
						<p>Slice Index: {record.slice_index}</p>

						{/* Hiển thị thời gian (timestamp) */}
						<p>Timestamp: {record.timestamp}</p>

						{/* Nút xóa lịch sử */}
						<button
							onClick={(e) => {
								e.stopPropagation(); // Ngăn sự kiện click vào thẻ
								handleDeleteHistory(record.id);
							}}
						>
							Xóa
						</button>
					</div>
				))}
			</div>
		</>
	);
}

export default App;
