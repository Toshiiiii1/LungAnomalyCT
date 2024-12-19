import { Outlet, Link } from "react-router-dom";

const Layout = () => {
	return (
		<>
			<nav>
				<ul>
					<li>
						<Link to="/">Lịch sử</Link>
					</li>
					<li>
						<Link to="/upload_mhdraw">Tải lên file MHD/RAW</Link>
					</li>
					<li>
						<Link to="/upload_image">Tải lên file PNG</Link>
					</li>
					<li>
						<Link to="/upload_dcm">Tải lên file DICOM</Link>
					</li>
				</ul>
			</nav>

			<Outlet />
		</>
	);
};

export default Layout;
