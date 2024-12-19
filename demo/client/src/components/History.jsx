import { useState, useEffect } from "react";
import axios from "axios";

const History = () => {
	const [result, setResult] = useState([]); // Danh sách ảnh từ API
	const [currentIndex, setCurrentIndex] = useState(0); // Ảnh hiện tại
	const [predictIndex, setPredictIndex] = useState(0); // Ảnh hiện tại
	const [predictedImageUrl, setPredictedImageUrl] = useState(""); // URL ảnh đã xử lý
	const [predictedImageUrls, setPredictedImageUrls] = useState([]);
	const [isLoading, setIsLoading] = useState(false);
	const [history, setHistory] = useState([]);
	const [selectedHistory, setSelectedHistory] = useState(null);
	const [fileType, setFileType] = useState("");

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
		setResult(record.original_images);
		setCurrentIndex(0);
		setPredictIndex(0);
		setPredictedImageUrl("");
		setPredictedImageUrls(record.predicted_images);
		setFileType(record.file_type);
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

	const handleImageClick = (url, sliceNumber) => {
		setPredictedImageUrl(url);
		if (fileType === "mhd/raw" || fileType === "dcm") {
			setCurrentIndex(sliceNumber);
			setPredictIndex(sliceNumber);
		}
		window.scrollTo({ top: 0});
	};

	return (
		<>
			<div className="app-container">
				{/* Sidebar */}
				<div className="sidebar">
					<h2>Lịch sử nhận diện</h2>
					{history.map((record) => (
						<div
							key={record.id}
							className="history-card"
							onClick={() => handleHistoryClick(record)}
						>
							<h3>Tên file/thư mục: {record.file_name}</h3>
							<p>Định dạng: {record.file_type}</p>
							<p>Thời gian: {record.timestamp}</p>
							<button
								onClick={(e) => {
									e.stopPropagation();
									handleDeleteHistory(record.id);
								}}
							>
								Xóa
							</button>
						</div>
					))}
				</div>

				{/* Main Content */}
				<div className="main-content">
					{isLoading ? (
						<p>Đang tải ảnh...</p>
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
										{fileType === "mhd/raw" ||
										fileType === "dcm"
											? `Lát cắt số ${
													currentIndex + 1
											  } trong số ${
													result.length
											  } lát cắt`
											: fileType === "png/jpg/jpeg"
											? "Ảnh gốc"
											: ""}
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
											{fileType === "mhd/raw" ||
											fileType === "dcm" ? (
												<>
													Hình ảnh dự đoán:{" "}
													{`Lát cắt số ${
														predictIndex + 1
													}`}
												</>
											) : fileType === "png/jpg/jpeg" ? (
												"Ảnh dự đoán"
											) : null}
										</p>
									</div>
								)}
							</div>

							{/* Các nút điều khiển Previous và Next */}
							{(fileType === "mhd/raw" || fileType === "dcm") && (
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
								</div>
							)}
							<div>
								<h2>Các ảnh dự đoán</h2>
								<div
									style={{
										display: "flex",
										flexWrap: "wrap",
										gap: "10px",
										justifyContent: "center",
									}}
								>
									{predictedImageUrls.length > 0 ? (
										predictedImageUrls.map((url, index) => {
											// Trích xuất số thứ tự mặt cắt từ URL
											const sliceNumberMatch =
												url.match(/_slice_(\d+)\.png$/);
											const sliceNumber = sliceNumberMatch
												? parseInt(
														sliceNumberMatch[1],
														10
												  )
												: "Unknown";

											return (
												<div
													key={index}
													style={{
														textAlign: "center",
													}}
												>
													<img
														src={url}
														alt={`Slice ${sliceNumber}`}
														style={{
															maxWidth: "300px",
															maxHeight: "300px",
															cursor: "pointer",
														}}
														onClick={() =>
															handleImageClick(
																url,
																sliceNumber
															)
														}
													/>
													<p>
														{sliceNumber !==
														"Unknown"
															? `Lát cắt số ${
																	sliceNumber +
																	1
															  }`
															: "Ảnh dự đoán"}
													</p>
												</div>
											);
										})
									) : (
										<p>Không có ảnh để hiển thị</p>
									)}
								</div>
							</div>
						</>
					) : (
						<p>Không có ảnh để hiển thị</p>
					)}
				</div>
			</div>
		</>
	);
};

export default History;
