import React from "react";
import clsx from "clsx";

const PALETTE = [
  "text-rose-500",
  "text-pink-500",
  "text-fuchsia-500",
  "text-purple-500",
  "text-blue-500",
  "text-sky-500",
  "text-cyan-500",
  "text-teal-500",
  "text-emerald-500",
  "text-lime-500",
  "text-yellow-500",
  "text-amber-500",
  "text-red-500",
];

/**
 * Convert an arbitrary string to a deterministic index into the colour palette.
 */
function stringToPaletteClass(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    // Simple hash function (djb2‑ish)
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  const index = Math.abs(hash) % PALETTE.length;
  return PALETTE[index];
}

export interface AvatarProps {
  /** Full nickname of the user */
  username: string;
  /** Pixel size of the avatar (both width & height). Default: 40px */
  size?: number;
  /** Extra utility classes */
  className?: string;
}

/**
 * <Avatar username="JongHoon" size={48} />
 */
export const Avatar: React.FC<AvatarProps> = ({
  username,
  size = 40,
  className,
}) => {
  const initial = username.trim().charAt(0).toUpperCase();
  const textClass = stringToPaletteClass(username);

  return (
    <span
      role="img"
      aria-label={`Avatar for ${username}`}
      className={clsx(
        textClass,
        "inline-flex items-center justify-center rounded-full select-none font-semibold uppercase",
        className
      )}
      style={{ width: size, height: size, fontSize: size * 0.5 }}
    >
      {initial}
    </span>
  );
};

export default Avatar;
