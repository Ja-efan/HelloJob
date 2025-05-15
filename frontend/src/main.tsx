import { Suspense } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { RouterProvider } from "react-router";
import router from "./routes/routes.tsx";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ErrorBoundary } from "react-error-boundary";
import Loading from "./components/Loading/Loading.tsx";
import RouterErrorHandler from "./components/Error/RouterErrorHandler.tsx";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      throwOnError: true,
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <ErrorBoundary FallbackComponent={RouterErrorHandler}>
    <QueryClientProvider client={queryClient}>
      <Suspense fallback={<Loading />}>
        <RouterProvider router={router} />
      </Suspense>
    </QueryClientProvider>
  </ErrorBoundary>
);
