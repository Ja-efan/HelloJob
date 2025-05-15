import React, { ReactNode } from "react";
import { Link } from "react-router";

interface LocateBtnProps {
  to: string;
  iconComponent: ReactNode;
  title: string;
  description: string;
}

const LocateBtn: React.FC<LocateBtnProps> = ({
  to,
  iconComponent,
  title,
  description,
}) => {
  return (
    <Link to={to}>
      <div className="w-56 h-56 bg-white rounded-t-xs rounded-b-md border border-[#E4E8F0] border-t-4 border-t-[#6F52E0] p-4 flex flex-col items-center shadow-md hover:shadow-none transition-all duration-150">
        <div className="w-22 h-22 bg-[#AF9BFF]/30 rounded-full flex justify-center items-center text-[#6F4BFF] mb-3">
          {iconComponent}
        </div>
        <h3 className="text-xl font-semibold mb-2">{title}</h3>
        <p className="text-[#6E7180] text-sm break-keep whitespace-normal text-center">
          {description}
        </p>
      </div>
    </Link>
  );
};

export default LocateBtn;
