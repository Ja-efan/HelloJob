interface FormInputProps {
  type:
    | "text"
    | "email"
    | "url"
    | "number"
    | "tel"
    | "data"
    | "range"
    | "password"
    | "date";
  width: string;
  height: string;
  name: string;
  className?: string;
  label?: string;
  require?: boolean;
  placeholder?: string;
  onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  value?: string;
}

function FormInput({
  type,
  name,
  className,
  label,
  require,
  width,
  height,
  placeholder,
  onChange,
  value,
}: FormInputProps) {
  return (
    <div className="my-2">
      {label && (
        <>
          <label htmlFor=""></label>
          <div className="text-foreground text-sm font-semibold mb-2">
            {label} {require && <span className="text-destructive">*</span>}
          </div>
        </>
      )}

      <input
        type={type}
        name={name}
        required={require === true}
        placeholder={placeholder}
        style={{ width, height }}
        value={value}
        className={`border-1 px-3 py-2 rounded-md outline-none not-placeholder-shown:invalid:ring not-placeholder-shown:invalid:border-none not-placeholder-shown:invalid:ring-destructive focus:ring-2 focus:ring-accent focus:border-none focus:invalid:ring-destructive focus:invalid:ring-2 ${className}`}
        onChange={onChange}
      />
    </div>
  );
}

export default FormInput;
