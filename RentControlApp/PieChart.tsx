import React, { useEffect, useState } from "react";
import { View, Text, Pressable, Animated } from "react-native";
import Svg, { Path, Circle, Text as SvgText } from "react-native-svg";
type Slice = { label: string; value: number; color: string; display?: string };
export default function PieChart({ rows }: { rows: Slice[] }) {
  const [selected, setSelected] = useState<string | null>(null);
  const [entrance] = useState(() => new Animated.Value(0));
  const dataKey = rows
    .map((row) => `${row.label}:${Number.isFinite(row.value) ? row.value : 0}`)
    .join("|");

  useEffect(() => {
    entrance.setValue(0.86);
    const animation = Animated.spring(entrance, {
      toValue: 1,
      useNativeDriver: false,
      speed: 14,
      bounciness: 3,
    });
    animation.start();
    return () => animation.stop();
  }, [dataKey, entrance]);

  const valueOf = (value: number) =>
    Number.isFinite(value) ? Math.max(0, value) : 0;
  const total = rows.reduce((sum, row) => sum + valueOf(row.value), 0);

  return (
    <View style={{ alignItems: "center", paddingTop: 18 }}>
      <Animated.View
        style={{
          opacity: entrance,
          transform: [
            {
              scale: entrance.interpolate({
                inputRange: [0, 1],
                outputRange: [0.8, 1],
              }),
            },
          ],
        }}
      >
        <Svg
          width={220}
          height={220}
          viewBox="0 0 220 220"
          accessibilityLabel="Distribución en gráfica de pastel"
        >
          {!total && <Circle cx={110} cy={110} r={96} fill="#e8efea" />}
          {rows.map((row, i) => {
              const value = valueOf(row.value);
              if (!value || !total) return null;

              const angle = (value / total) * Math.PI * 2;
              const consumed = rows
                .slice(0, i)
                .reduce((sum, item) => sum + valueOf(item.value), 0);
              const start =
                -Math.PI / 2 + (consumed / total) * Math.PI * 2;
              const end = start + angle;

              const x = (a: number) => 110 + 96 * Math.cos(a);
              const y = (a: number) => 110 + 96 * Math.sin(a);
              const percentage = (value / total) * 100;
              const mid = start + angle / 2;

              return (
                <React.Fragment key={row.label}>
                  {value === total ? (
                    <Circle
                      cx={110}
                      cy={110}
                      r={96}
                      fill={row.color}
                      onPress={() =>
                        setSelected(selected === row.label ? null : row.label)
                      }
                    />
                  ) : (
                    <Path
                      d={`M 110 110 L ${x(start)} ${y(start)} A 96 96 0 ${angle > Math.PI ? 1 : 0} 1 ${x(end)} ${y(end)} Z`}
                      fill={row.color}
                      stroke="#fff"
                      strokeWidth={3}
                      opacity={
                        selected === null || selected === row.label ? 1 : 0.4
                      }
                      onPress={() =>
                        setSelected(selected === row.label ? null : row.label)
                      }
                    />
                  )}
                  {percentage >= 8 && (
                    <SvgText
                      x={110 + 59 * Math.cos(mid)}
                      y={110 + 59 * Math.sin(mid)}
                      fill="#fff"
                      fontSize={12}
                      fontWeight="700"
                      textAnchor="middle"
                      alignmentBaseline="middle"
                      pointerEvents="none"
                    >
                      {percentage.toFixed(0)}%
                    </SvgText>
                  )}
                </React.Fragment>
              );
            })}
        </Svg>
      </Animated.View>
      <Text style={{ fontSize: 13, color: "#658272", marginBottom: 15 }}>
        {total
          ? "Toca una categoría para ver su proporción"
          : "Aún no hay datos para graficar"}
      </Text>
      {rows.map((row, i) => (
        <Pressable
          key={row.label}
          accessibilityRole="button"
          onPress={() =>
            setSelected(selected === row.label ? null : row.label)
          }
          style={{
            flexDirection: "row",
            gap: 9,
            alignItems: "center",
            padding: 10,
            borderRadius: 8,
            backgroundColor:
              selected === row.label ? "#edf5ef" : "transparent",
            width: "100%",
          }}
        >
          <View
            style={{
              width: 10,
              height: 10,
              borderRadius: 4,
              backgroundColor: row.color,
            }}
          />
          <Text style={{ flex: 1, color: "#43634f", fontSize: 13 }}>
            {row.label}
          </Text>
          <Text style={{ color: "#1d5036", fontWeight: "700" }}>
            {total ? ((valueOf(row.value) / total) * 100).toFixed(1) : "0"}%
          </Text>
          <Text style={{ color: "#658272", fontSize: 12 }}>
            {row.display ?? row.value}
          </Text>
        </Pressable>
      ))}
    </View>
  );
}
