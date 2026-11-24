// Bin state from the fill percentage. Cut-offs: 80% and above is Full,
// 20% up to 80% is Half-full, below 20% is Empty.
export const fillState = (percent) => {
  const value = Number(percent) || 0;
  if (value >= 80) return { label: "Full", color: "#dc2626" };
  if (value >= 20) return { label: "Half-full", color: "#d97706" };
  return { label: "Empty", color: "#059669" };
};
