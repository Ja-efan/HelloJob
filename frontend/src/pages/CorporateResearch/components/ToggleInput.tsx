import { UseFormRegister } from "react-hook-form";

interface ToggleInputProps {
  label: string;
  description: string;
  isOn: boolean;
  onChange: (value: boolean) => void;
  register: UseFormRegister<any>;
  name: string;
  required?: boolean;
  requiredMessage?: string;
  disabled?: boolean;
  isDisabledReason?: string;
}

function ToggleInput({
  label,
  description,
  isOn,
  onChange,
  register,
  name,
  required = false,
  requiredMessage = "필수 입력 항목입니다.",
  disabled,
  isDisabledReason = "Dart 미지원",
}: ToggleInputProps) {
  return (
    <div className="flex items-center p-4 bg-white rounded-lg shadow-sm border border-[#E4E8F0]">
      <div className="flex-1">
        <div className="flex flex-row items-center mb-1">
          <span
            className={`text-lg font-bold mb-1 block ${
              isOn ? "text-[#6F52E0]" : "text-[#6E7180]"
            }`}
          >
            {label}
          </span>
          {disabled && isDisabledReason && (
            <span className="text-sm text-gray-400 ml-2">
              {isDisabledReason}
            </span>
          )}
        </div>
        <p className="text-base text-gray-950">{description}</p>
      </div>
      <label
        className={`flex items-center cursor-pointer ${
          disabled ? "opacity-50 cursor-not-allowed" : ""
        }`}
      >
        <input
          type="checkbox"
          className="sr-only peer"
          {...register(name, {
            required: required ? requiredMessage : false,
          })}
          checked={isOn}
          disabled={disabled}
          onChange={() => {
            if (disabled) return;
            // onChange(e.target.checked);
            onChange(!isOn);
          }}
        />
        <div
          className={`relative w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-purple-300 dark:peer-focus:ring-purple-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-[#886BFB] dark:peer-checked:bg-[#886BFB]`}
        ></div>
      </label>
    </div>
  );
}

export default ToggleInput;
