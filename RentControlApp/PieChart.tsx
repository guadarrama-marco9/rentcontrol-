import React, { useEffect, useState } from "react";
import { View, Text, Pressable, Animated } from "react-native";
import Svg, { Path, Circle } from "react-native-svg";
type Slice = { label: string; value: number; color: string; display?: string };
export default function PieChart({ rows }: { rows: Slice[] }) {
  const [selected, setSelected] = useState<number | null>(null);
  const [entrance] = useState(() => new Animated.Value(0));
  useEffect(() => {
    Animated.timing(entrance, {
      toValue: 1,
      duration: 650,
      useNativeDriver: true,
    }).start();
  }, [entrance]);
  const total = rows.reduce((s, r) => s + Math.max(0, r.value), 0);

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
            const value = Math.max(0, row.value);
            if (!value || !total) return null;
            const angle = (value / total) * Math.PI * 2,
              start = -Math.PI / 2 + rows.slice(0,i).reduce((sum,item)=>sum+Math.max(0,item.value),0)/total*Math.PI*2;
            const end = start + angle;
            const x = (a: number) => 110 + 96 * Math.cos(a),
              y = (a: number) => 110 + 96 * Math.sin(a);
            return value === total ? (
              <Circle
                key={i}
                cx={110}
                cy={110}
                r={96}
                fill={row.color}
                onPress={() => setSelected(i)}
              />
            ) : (
              <Path
                key={i}
                d={`M 110 110 L ${x(start)} ${y(start)} A 96 96 0 ${angle > Math.PI ? 1 : 0} 1 ${x(end)} ${y(end)} Z`}
                fill={row.color}
                stroke="#fff"
                strokeWidth={3}
                opacity={selected === null || selected === i ? 1 : 0.4}
                onPress={() => setSelected(selected === i ? null : i)}
              />
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
          onPress={() => setSelected(selected === i ? null : i)}
          style={{
            flexDirection: "row",
            gap: 9,
            alignItems: "center",
            padding: 10,
            borderRadius: 8,
            backgroundColor: selected === i ? "#edf5ef" : "transparent",
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
            {total ? ((Math.max(0, row.value) / total) * 100).toFixed(1) : "0"}%
          </Text>
          <Text style={{ color: "#658272", fontSize: 12 }}>
            {row.display ?? row.value}
          </Text>
        </Pressable>
      ))}
    </View>
  );
}
