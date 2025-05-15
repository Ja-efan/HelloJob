import { Outlet } from "react-router";

function MainLayout() {
  return (
    <main className="grow max-w-screen-xl mx-auto w-full">
      <Outlet />
    </main>
  );
}

export default MainLayout;
