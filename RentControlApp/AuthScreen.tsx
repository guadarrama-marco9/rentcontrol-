import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  TextInput,
  Pressable,
  Animated,
  ScrollView,
  StyleSheet,
  useWindowDimensions,
  ActivityIndicator,
} from "react-native";

type Props = {
  request: (path: string, body?: any) => Promise<any>;
  onLogin: (token: string) => Promise<void>;
};

export default function AuthScreen({
  request,
  onLogin,
}: Props) {
  const { width } = useWindowDimensions();

  const [register, setRegister] = useState(false);
  const [portal, setPortal] = useState<"admin" | "user">("admin");
  const [motion] = useState(() => new Animated.Value(0));

  const [setup, setSetup] = useState<boolean | null>(null);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const animation = Animated.loop(
      Animated.sequence([
        Animated.timing(motion, {
          toValue: 1,
          duration: 7000,
          useNativeDriver: true,
        }),
        Animated.timing(motion, {
          toValue: 0,
          duration: 7000,
          useNativeDriver: true,
        }),
      ]),
    );

    animation.start();

    return () => animation.stop();
  }, [motion]);

  useEffect(() => {
    let active = true;

    request("/auth/status")
      .then((result) => {
        if (active) setSetup(result.setup_required);
      })
      .catch((e) => {
        if (active) setError(e.message);
      });

    return () => {
      active = false;
    };
  }, [request]);

  async function retry() {
    setError("");

    try {
      const result = await request("/auth/status");
      setSetup(result.setup_required);
    } catch (e: any) {
      setError(e.message);
    }
  }

  async function submit() {
    if (busy || setup === null) return;

    setBusy(true);
    setError("");

    try {
      if (!email.trim()) {
        throw Error("Escribe tu correo electrónico.");
      }

      if (password.length < 10) {
        throw Error(
          "Usa una contraseña de al menos 10 caracteres.",
        );
      }

      if ((setup || register) && name.trim().length < 2) {
        throw Error("Escribe tu nombre.");
      }

      if (setup) {
        await request("/auth/setup", {
          name,
          email,
          password,
        });
      } else if (register) {
        await request("/auth/register", {
          name,
          email,
          password,
        });
      }

      const result = await request("/auth/login", {
        email,
        password,
      });

      if (!setup && !register && result.role !== portal) {
        throw Error(
          result.role === "admin"
            ? "Esta cuenta es de administrador. Selecciona Administrador."
            : "Esta cuenta es de usuario. Selecciona Usuario.",
        );
      }

      await onLogin(result.access_token);
    } catch (e: any) {
      setError(e.message);

      if (setup) {
        request("/auth/status")
          .then((result) => {
            setSetup(result.setup_required);
          })
          .catch(() => {
            // Conserva el mensaje del error original.
          });
      }
    } finally {
      setBusy(false);
    }
  }

  const title = setup
    ? "Crear administrador"
    : register
      ? "Crear cuenta de usuario"
      : "Qué gusto verte";

  const description = setup
    ? "Configura al primer administrador de RentControl."
    : register
      ? "Consultarás únicamente los inmuebles que te asigne el administrador."
      : "Selecciona tu acceso e inicia sesión.";

  const buttonText = setup
    ? "Crear administrador y entrar →"
    : register
      ? "Crear cuenta de usuario →"
      : portal === "admin"
        ? "Entrar como administrador →"
        : "Entrar como usuario →";

  return (
    <View style={s.root}>
      <View
        pointerEvents="none"
        style={StyleSheet.absoluteFill}
      >
        <Animated.View
          style={[
            s.orb,
            {
              top: -180,
              left: -100,
              backgroundColor: "#247b63",
              transform: [
                {
                  translateX: motion.interpolate({
                    inputRange: [0, 1],
                    outputRange: [0, 100],
                  }),
                },
                {
                  translateY: motion.interpolate({
                    inputRange: [0, 1],
                    outputRange: [0, 80],
                  }),
                },
              ],
            },
          ]}
        />

        <Animated.View
          style={[
            s.orb,
            {
              bottom: -230,
              right: -100,
              backgroundColor: "#215e61",
              transform: [
                {
                  translateX: motion.interpolate({
                    inputRange: [0, 1],
                    outputRange: [0, -100],
                  }),
                },
                {
                  scale: motion.interpolate({
                    inputRange: [0, 1],
                    outputRange: [1, 1.3],
                  }),
                },
              ],
            },
          ]}
        />

        {[0, 1, 2, 3, 4].map((i) => (
          <Animated.View
            key={i}
            style={{
              position: "absolute",
              width: 6 + i * 2,
              height: 6 + i * 2,
              borderRadius: 20,
              backgroundColor: "#9ce1c4",
              opacity: 0.3,
              left: `${12 + i * 17}%`,
              top: `${15 + i * 15}%`,
              transform: [
                {
                  translateY: motion.interpolate({
                    inputRange: [0, 1],
                    outputRange: [0, -35 - i * 8],
                  }),
                },
              ],
            }}
          />
        ))}
      </View>

      <ScrollView
        contentContainerStyle={{
          flexGrow: 1,
          justifyContent: "center",
          padding: width > 850 ? 65 : 22,
        }}
        keyboardShouldPersistTaps="handled"
      >
        <View
          style={{
            width: "100%",
            maxWidth: 1120,
            alignSelf: "center",
            flexDirection: width > 850 ? "row" : "column",
            alignItems: "center",
            gap: width > 850 ? 70 : 30,
          }}
        >
          <View style={{ flex: 1 }}>
            <Text style={s.brand}>
              rent
              <Text style={{ color: "#75e7bb" }}>
                control.
              </Text>
            </Text>

            <Text style={s.tag}>
              TU ESPACIO. TU CONTROL.
            </Text>

            <Text
              style={[
                s.title,
                {
                  fontSize: width > 850 ? 57 : 36,
                  lineHeight: width > 850 ? 65 : 44,
                },
              ]}
            >
              Tu patrimonio,{"\n"}
              en buenas manos.
            </Text>

            <Text style={s.subtitle}>
              Menos pendientes. Más tranquilidad.
              Administra tus propiedades, cobros y
              mantenimientos desde un mismo lugar.
            </Text>

            <View style={s.chips}>
              {[
                "⌂ Propiedades",
                "↗ Cobranza",
                "✦ RentIA",
              ].map((text) => (
                <Text key={text} style={s.chip}>
                  {text}
                </Text>
              ))}
            </View>
          </View>

          <View style={s.panel}>
            <Text style={s.panelTitle}>{title}</Text>

            <Text style={s.muted}>{description}</Text>

            {!setup && !register && setup !== null && (
              <View style={s.roles}>
                {(["admin", "user"] as const).map((role) => (
                  <Pressable
                    key={role}
                    accessibilityRole="button"
                    disabled={busy}
                    onPress={() => {
                      setPortal(role);
                      setError("");
                    }}
                    style={[
                      s.roleButton,
                      portal === role && s.roleActive,
                    ]}
                  >
                    <Text
                      style={[
                        s.roleText,
                        portal === role && {
                          color: "#fff",
                        },
                      ]}
                    >
                      {role === "admin"
                        ? "Administrador"
                        : "Usuario"}
                    </Text>
                  </Pressable>
                ))}
              </View>
            )}

            {(setup || register) && (
              <>
                <Text style={s.label}>Tu nombre</Text>

                <TextInput
                  accessibilityLabel="Tu nombre"
                  value={name}
                  onChangeText={setName}
                  editable={!busy}
                  style={s.input}
                  maxLength={100}
                  placeholder="Nombre completo"
                  placeholderTextColor="#91a297"
                />
              </>
            )}

            <Text style={s.label}>
              Correo electrónico
            </Text>

            <TextInput
              accessibilityLabel="Correo electrónico"
              value={email}
              onChangeText={setEmail}
              editable={!busy}
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="email-address"
              style={s.input}
              maxLength={200}
              placeholder="correo@ejemplo.com"
              placeholderTextColor="#91a297"
            />

            <Text style={s.label}>Contraseña</Text>

            <View style={s.passwordRow}>
              <TextInput
                accessibilityLabel="Contraseña"
                value={password}
                onChangeText={setPassword}
                editable={!busy}
                secureTextEntry={!show}
                autoCapitalize="none"
                autoCorrect={false}
                style={[s.input, { flex: 1 }]}
                maxLength={128}
                onSubmitEditing={submit}
                placeholder="Mínimo 10 caracteres"
                placeholderTextColor="#91a297"
              />

              <Pressable
                accessibilityRole="button"
                onPress={() => setShow(!show)}
                style={{ padding: 10 }}
              >
                <Text style={{ color: "#187859" }}>
                  {show ? "Ocultar" : "Ver"}
                </Text>
              </Pressable>
            </View>

            <Text style={s.passwordHint}>
              Tu contraseña se guarda protegida.
            </Text>

            {!!error && (
              <Text
                accessibilityRole="alert"
                style={s.error}
              >
                {error}
              </Text>
            )}

            {setup === null ? (
              <Pressable
                accessibilityRole="button"
                onPress={retry}
                style={s.button}
              >
                <Text style={s.buttonText}>
                  {error
                    ? "Reintentar conexión"
                    : "Conectando…"}
                </Text>
              </Pressable>
            ) : (
              <Pressable
                accessibilityRole="button"
                disabled={busy}
                onPress={submit}
                style={[
                  s.button,
                  busy && { opacity: 0.6 },
                ]}
              >
                {busy ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <Text style={s.buttonText}>
                    {buttonText}
                  </Text>
                )}
              </Pressable>
            )}

            {!setup && setup !== null && (
              <Pressable
                accessibilityRole="button"
                disabled={busy}
                onPress={() => {
                  setRegister(!register);
                  setPortal("user");
                  setError("");
                }}
                style={s.registerButton}
              >
                <Text style={s.registerText}>
                  {register
                    ? "Ya tengo cuenta. Iniciar sesión"
                    : "¿Eres inquilino? Crear cuenta de usuario"}
                </Text>
              </Pressable>
            )}

            <Text style={s.footer}>
              El administrador asigna los inmuebles
              que puede consultar cada usuario.
            </Text>
          </View>
        </View>
      </ScrollView>
    </View>
  );
}

const s = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: "#0d3027",
    overflow: "hidden",
  },
  orb: {
    position: "absolute",
    width: 650,
    height: 650,
    borderRadius: 400,
    opacity: 0.4,
  },
  brand: {
    fontSize: 33,
    color: "#fff",
    fontWeight: "800",
    letterSpacing: -1,
    marginBottom: 55,
  },
  tag: {
    fontSize: 12,
    color: "#89dfbb",
    letterSpacing: 3,
    fontWeight: "700",
  },
  title: {
    fontWeight: "800",
    color: "#fff",
    letterSpacing: -2,
    marginVertical: 18,
  },
  subtitle: {
    color: "#bbd5c9",
    fontSize: 17,
    lineHeight: 28,
    maxWidth: 460,
  },
  chips: {
    flexDirection: "row",
    gap: 12,
    flexWrap: "wrap",
    marginTop: 30,
  },
  chip: {
    paddingVertical: 11,
    paddingHorizontal: 15,
    borderWidth: 1,
    borderColor: "#4c7e69",
    borderRadius: 30,
    color: "#d0e8da",
  },
  panel: {
    width: "100%",
    maxWidth: 425,
    backgroundColor: "#fff",
    padding: 32,
    borderRadius: 25,
  },
  panelTitle: {
    fontSize: 29,
    color: "#173f30",
    fontWeight: "800",
    marginBottom: 8,
  },
  muted: {
    fontSize: 14,
    lineHeight: 22,
    color: "#7a8d82",
  },
  roles: {
    flexDirection: "row",
    gap: 10,
    marginTop: 20,
  },
  roleButton: {
    flex: 1,
    padding: 13,
    borderRadius: 10,
    backgroundColor: "#edf4ef",
  },
  roleActive: {
    backgroundColor: "#218761",
  },
  roleText: {
    textAlign: "center",
    fontWeight: "700",
    color: "#345748",
  },
  label: {
    fontSize: 13,
    fontWeight: "600",
    color: "#345748",
    marginTop: 20,
    marginBottom: 9,
  },
  input: {
    backgroundColor: "#f3f7f4",
    borderWidth: 1,
    borderColor: "#dfebe3",
    borderRadius: 11,
    padding: 14,
    color: "#224533",
    fontSize: 15,
  },
  passwordRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  passwordHint: {
    fontSize: 12,
    lineHeight: 20,
    color: "#7a8d82",
    marginTop: 7,
  },
  button: {
    backgroundColor: "#218761",
    padding: 17,
    borderRadius: 12,
    marginTop: 25,
    alignItems: "center",
  },
  buttonText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: 15,
    textAlign: "center",
  },
  registerButton: {
    padding: 15,
    marginTop: 10,
  },
  registerText: {
    textAlign: "center",
    color: "#187859",
    fontWeight: "600",
  },
  error: {
    color: "#b84439",
    marginTop: 15,
    lineHeight: 21,
  },
  footer: {
    color: "#7a8d82",
    textAlign: "center",
    fontSize: 12,
    lineHeight: 20,
    marginTop: 20,
  },
});