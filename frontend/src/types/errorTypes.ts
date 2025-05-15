export interface RenderErrorHandlerProps {
  error: Error;
  resetErrorBoundary: () => void;
}

export interface ErrorPageProps {
  status?: number | string;
  title: string;
  message?: string;
  actionButton?: {
    label: string;
    onClick: () => void;
  };
  showHomeButton?: boolean;
  showBackButton?: boolean;
}
