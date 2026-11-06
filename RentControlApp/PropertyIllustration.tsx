import React from 'react';
import { Text, View } from 'react-native';
import Svg, { Circle, Ellipse, Path, Rect } from 'react-native-svg';

const palettes = [
  {
    sky: '#dcefe9',
    hill: '#b5d8c6',
    lawn: '#8ebc98',
    wall: '#f5ead7',
    roof: '#446e5d',
    trim: '#d6aa73',
    door: '#815f47',
  },
  {
    sky: '#e9e5d8',
    hill: '#c6d5bd',
    lawn: '#a0bc8d',
    wall: '#f1f0e8',
    roof: '#8b6555',
    trim: '#d7bd92',
    door: '#557866',
  },
  {
    sky: '#dce8f0',
    hill: '#c0d5d0',
    lawn: '#91b59e',
    wall: '#e8d9c8',
    roof: '#536777',
    trim: '#ead7ab',
    door: '#9b6e52',
  },
];

export default function PropertyIllustration({
  seed,
  label,
}: {
  seed: string;
  label: string;
}) {
  const variant = Array.from(seed).reduce(
    (sum, character) => sum + character.charCodeAt(0),
    0,
  ) % palettes.length;
  const colors = palettes[variant];

  return (
    <View
      accessible
      accessibilityLabel={`Ilustración de una casa para ${label}`}
      style={{
        height: 180,
        overflow: 'hidden',
        borderRadius: 13,
        marginBottom: 14,
        backgroundColor: colors.sky,
      }}
    >
      <Svg
        width="100%"
        height="100%"
        viewBox="0 0 700 260"
        preserveAspectRatio="xMidYMid slice"
      >
        <Rect width="700" height="260" fill={colors.sky} />
        <Circle cx="570" cy="58" r="26" fill="#f3d996" />
        <Path
          d="M0 166 Q115 102 238 161 T477 149 T700 155 V260 H0Z"
          fill={colors.hill}
        />
        <Rect y="190" width="700" height="70" fill={colors.lawn} />
        <Ellipse cx="350" cy="218" rx="178" ry="14" fill="#658f72" opacity={0.28} />

        <Path d="M213 134 347 55 490 134Z" fill={colors.roof} />
        <Rect x="237" y="130" width="220" height="83" fill={colors.wall} />
        <Rect x="253" y="145" width="43" height="47" fill="#a8d1d1" />
        <Rect x="397" y="145" width="43" height="47" fill="#a8d1d1" />
        <Rect x="329" y="151" width="43" height="62" fill={colors.door} />
        <Rect x="318" y="145" width="65" height="7" fill={colors.trim} />
        <Rect x="248" y="189" width="181" height="8" fill={colors.trim} />
        <Rect x="267" y="162" width="15" height="4" fill="#f8f3e5" opacity={0.8} />
        <Rect x="411" y="162" width="15" height="4" fill="#f8f3e5" opacity={0.8} />

        <Path d="M505 194V116h38v78" fill="#789477" />
        <Circle cx="524" cy="105" r="30" fill="#729779" />
        <Circle cx="507" cy="120" r="20" fill="#8bb18a" />
        <Circle cx="540" cy="120" r="22" fill="#81a981" />
        <Path d="M558 198V145h25v53" fill="#789477" />
        <Circle cx="570" cy="137" r="23" fill="#709377" />
        <Circle cx="555" cy="148" r="16" fill="#8bb18a" />
        <Path d="M153 200V157h22v43" fill="#789477" />
        <Circle cx="164" cy="148" r="22" fill="#719577" />
        <Circle cx="150" cy="158" r="15" fill="#89ad84" />
      </Svg>
      <View
        style={{
          position: 'absolute',
          left: 10,
          bottom: 10,
          borderRadius: 12,
          backgroundColor: 'rgba(255,255,255,0.9)',
          paddingHorizontal: 10,
          paddingVertical: 5,
        }}
      >
        <Text style={{ color: '#365c48', fontSize: 11, fontWeight: '700' }}>
          Imagen ilustrativa
        </Text>
      </View>
    </View>
  );
}
