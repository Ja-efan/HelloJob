import { create } from "zustand";

interface AuthState {
  isLoggedIn: boolean;
  userName: string;
  isLoginAlert: boolean;
  setIsLoggedIn: (status: boolean) => void;
  setUserName: (name: string) => void;
  setIsLoginAlert: (status: boolean) => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  isLoggedIn: false,
  userName: "",
  isLoginAlert: false,
  setIsLoggedIn: (status) => set({ isLoggedIn: status }),
  setUserName: (name) => set({ userName: name }),
  setIsLoginAlert: (status) => set({ isLoginAlert: status }),
}));
