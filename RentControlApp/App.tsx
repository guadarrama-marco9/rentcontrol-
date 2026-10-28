import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Animated,
  Linking,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
  useWindowDimensions,
} from 'react-native';

import AuthScreen from './AuthScreen';
import PieChart from './PieChart';

const API =
  process.env.EXPO_PUBLIC_API_URL || 'http://localhost:8000';

let token = '';

function setToken(value: string) {
  token = value;
}

async function api(path: string, body?: any) {
  const response = await fetch(API + path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: token ? 'Bearer ' + token : '',
      'X-API-Key': process.env.EXPO_PUBLIC_API_KEY || '',
    },
    ...(body === undefined
      ? {}
      : { body: JSON.stringify(body) }),
  });

  const result = await response.json();

  if (!response.ok) {
    throw Error(
      typeof result.detail === 'string'
        ? result.detail
        : 'Revisa los datos del formulario.',
    );
  }

  return result;
}

const money = (value: number) =>
  new Intl.NumberFormat('es-MX', {
    style: 'currency',
    currency: 'MXN',
  }).format(value / 100);

const today = () =>
  new Date().toLocaleDateString('en-CA', {
    timeZone: 'America/Mexico_City',
  });

function cents(value: string) {
  if (
    !/^\d+(\.\d{1,2})?$/.test(value.trim()) ||
    Number(value) <= 0
  ) {
    throw Error(
      'Escribe un importe positivo con máximo dos decimales.',
    );
  }

  return Math.round(Number(value) * 100);
}

function Button({
  title,
  onPress,
  secondary = false,
  disabled = false,
}: {
  title: string;
  onPress: () => void;
  secondary?: boolean;
  disabled?: boolean;
}) {
  const [scale] = useState(
    () => new Animated.Value(1),
  );

  function animate(value: number) {
    Animated.spring(scale, {
      toValue: value,
      useNativeDriver: true,
      speed: 35,
    }).start();
  }

  return (
    <Animated.View
      style={{
        alignSelf: 'flex-start',
        transform: [{ scale }],
      }}
    >
      <Pressable
        accessibilityRole="button"
        disabled={disabled}
        onPress={onPress}
        onPressIn={() => animate(0.96)}
        onPressOut={() => animate(1)}
        onHoverIn={() => animate(1.02)}
        onHoverOut={() => animate(1)}
        style={[
          s.button,
          secondary && s.secondary,
          disabled && { opacity: 0.5 },
        ]}
      >
        <Text
          style={{
            color: secondary ? '#285641' : '#fff',
            fontWeight: '700',
            fontSize: 13,
          }}
        >
          {title}
        </Text>
      </Pressable>
    </Animated.View>
  );
}

function Card({
  children,
}: {
  children: React.ReactNode;
}) {
  const [fade] = useState(
    () => new Animated.Value(0),
  );

  useEffect(() => {
    const animation = Animated.timing(fade, {
      toValue: 1,
      duration: 350,
      useNativeDriver: true,
    });

    animation.start();
    return () => animation.stop();
  }, [fade]);

  return (
    <Animated.View
      style={[
        s.card,
        {
          opacity: fade,
          transform: [
            {
              translateY: fade.interpolate({
                inputRange: [0, 1],
                outputRange: [10, 0],
              }),
            },
          ],
        },
      ]}
    >
      {children}
    </Animated.View>
  );
}

function Field({
  label,
  value,
  onChange,
  numeric = false,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  numeric?: boolean;
}) {
  return (
    <View style={{ marginBottom: 12 }}>
      <Text style={s.label}>{label}</Text>

      <TextInput
        accessibilityLabel={label}
        value={value}
        onChangeText={onChange}
        keyboardType={numeric ? 'decimal-pad' : 'default'}
        style={s.input}
      />
    </View>
  );
}

function Badge({
  text,
  warn = false,
}: {
  text: string;
  warn?: boolean;
}) {
  return (
    <Text
      style={[
        s.badge,
        warn && {
          backgroundColor: '#fff0e3',
          color: '#a65720',
        },
      ]}
    >
      {text}
    </Text>
  );
}

const menu = [
  'Inicio',
  'Propiedades',
  'Contratos',
  'Cobranza',
  'Mantenimientos',
  'Alertas',
  'Movimientos',
  'RentIA',
  'Usuarios',
  'Cómo se calculó',
  'Ayuda',
  'Laboratorio',
];

export default function App() {
  const { width } = useWindowDimensions();

  const [user, setUser] = useState<any>(null);
  const [data, setData] = useState<any>(null);
  const [page, setPage] = useState('Inicio');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);

  const [form, setForm] = useState<any>(null);
  const [values, setValues] = useState<any>({});
  const [users, setUsers] = useState<any[]>([]);
  const [explanation, setExplanation] =
    useState<any>(null);
  const [lab, setLab] = useState<any>(null);

  const [selected, setSelected] = useState('');
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState('Todos');

  const [help, setHelp] = useState('Guía rápida');
  const [calculation, setCalculation] =
    useState('Alertas');
  const [showTable, setShowTable] = useState(false);

  const [question, setQuestion] = useState(
    '¿Qué pendientes debo atender hoy?',
  );
  const [answer, setAnswer] = useState('');

  const admin = user?.role === 'admin';
  const properties = data?.properties || [];
  const metrics = data?.metrics;

  const charges = properties.flatMap((p: any) =>
    p.charges.map((c: any) => ({
      ...c,
      property: p,
    })),
  );

  const tasks = properties.flatMap((p: any) =>
    p.maintenance.map((m: any) => ({
      ...m,
      property: p,
    })),
  );

  const payments = properties
    .flatMap((p: any) =>
      (p.payments || []).map((m: any) => ({
        ...m,
        property: p,
      })),
    )
    .sort((a: any, b: any) =>
      b.date.localeCompare(a.date),
    );

  async function login(value: string) {
    setToken(value);

    try {
      setUser(await api('/auth/me'));
    } catch (e) {
      setToken('');
      throw e;
    }
  }

  async function logout() {
    try {
      await api('/auth/logout', {});
    } catch {
      // Permite volver al acceso si la sesión caducó.
    }

    setToken('');
    setUser(null);
    setData(null);
    setExplanation(null);
    setUsers([]);
    setLab(null);
    setForm(null);
    setAnswer('');
    setPage('Inicio');
    setError('');
  }

  async function refresh() {
    try {
      setData(await api('/dashboard'));
      setError('');
    } catch (e: any) {
      setError(e.message);
    }
  }

  useEffect(() => {
    if (!user) return;

    let active = true;

    api('/dashboard')
      .then((result) => {
        if (active) setData(result);
      })
      .catch((e) => {
        if (active) setError(e.message);
      });

    return () => {
      active = false;
    };
  }, [user]);

  useEffect(() => {
    if (!user) return;

    let active = true;

    const path =
      page === 'Usuarios' && admin
        ? '/users'
        : page === 'Cómo se calculó'
          ? '/explanations'
          : '';

    if (path) {
      api(path)
        .then((result) => {
          if (!active) return;

          if (path === '/users') {
            setUsers(result);
          } else {
            setExplanation(result);
          }
        })
        .catch((e) => {
          if (active) setError(e.message);
        });
    }

    return () => {
      active = false;
    };
  }, [page, user, admin]);

  function navigate(name: string) {
    setPage(name);
    setError('');
    setNotice('');
    setQuery('');
    setFilter('Todos');
    setForm(null);
  }

  function open(
    type: string,
    property?: any,
    charge?: any,
  ) {
    setError('');
    setNotice('');
    setForm({ type, property, charge });

    setValues({
      name: '',
      address: '',
      rent: property?.monthly_rent_cents
        ? String(property.monthly_rent_cents / 100)
        : '',
      tenant: '',
      start: today(),
      end: '',
      due: today(),
      amount: charge
        ? String(
            (charge.amount_cents - charge.paid_cents) /
              100,
          )
        : '',
      reference: '',
      description: '',
      urgent: false,
      user_id: property?.assigned_user_id,
    });
  }

  const field = (key: string) => (value: string) =>
    setValues((old: any) => ({
      ...old,
      [key]: value,
    }));

  async function save() {
    if (busy) return;

    setBusy(true);
    setError('');

    try {
      const base =
        '/properties/' + form.property?.id;

      if (form.type === 'property') {
        await api('/properties', {
          name: values.name,
          address: values.address,
          monthly_rent_cents: cents(values.rent),
        });
      }

      if (form.type === 'contract') {
        await api(base + '/contract', {
          tenant: values.tenant,
          start: values.start,
          end: values.end,
          monthly_rent_cents: cents(values.rent),
        });
      }

      if (form.type === 'charge') {
        await api(base + '/charges', {
          due: values.due,
          amount_cents: cents(values.rent),
        });
      }

      if (form.type === 'payment') {
        await api(
          base +
            '/charges/' +
            form.charge.id +
            '/payments',
          {
            amount_cents: cents(values.amount),
            reference: values.reference,
          },
        );
      }

      if (form.type === 'maintenance') {
        await api(base + '/maintenance', {
          description: values.description,
          urgent: values.urgent,
        });
      }

      if (form.type === 'access') {
        if (!values.user_id) {
          throw Error(
            'Selecciona una cuenta de usuario.',
          );
        }

        await api(base + '/access', {
          user_id: values.user_id,
        });
      }

      setForm(null);
      setNotice('Registro guardado correctamente.');
      await refresh();
      setExplanation(null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function resolve(task: any) {
    setBusy(true);

    try {
      await api(
        `/properties/${task.property.id}/maintenance/${task.id}/resolve`,
        {},
      );

      await refresh();
      setNotice('Mantenimiento resuelto.');
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function loadLab() {
    setBusy(true);

    try {
      const [result, logic] = await Promise.all([
        api('/lab'),
        api('/logic'),
      ]);

      setLab({ ...result, logic });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function ask() {
    setBusy(true);
    setError('');

    try {
      const result = await api('/agent', {
        message: question,
      });

      setAnswer(result.answer);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const heading = (
    title: string,
    subtitle: string,
  ) => (
    <View style={{ marginBottom: 20 }}>
      <Text style={s.title}>{title}</Text>
      <Text style={s.muted}>{subtitle}</Text>
    </View>
  );

  function showProperty(p: any) {
    setSelected(p.id);
    navigate('Propiedades');
  }

  function chargeCard(c: any) {
    const balance =
      c.amount_cents - c.paid_cents;

    return (
      <Card key={c.id}>
        <View style={s.between}>
          <Text style={s.h2}>
            {c.property.name}
          </Text>

          <Badge
            warn={
              balance > 0 &&
              c.due < data.as_of
            }
            text={
              balance === 0
                ? 'Liquidado'
                : c.due < data.as_of
                  ? 'Vencido'
                  : 'Pendiente'
            }
          />
        </View>

        <Text style={s.muted}>
          Vence: {c.due} · Cargo:{' '}
          {money(c.amount_cents)}
        </Text>

        <Text style={s.stat}>
          {money(balance)}
        </Text>

        <Text style={s.muted}>
          Saldo pendiente · Pagado:{' '}
          {money(c.paid_cents)}
        </Text>

        {admin && balance > 0 && (
          <Button
            title="Registrar pago"
            onPress={() =>
              open('payment', c.property, c)
            }
          />
        )}
      </Card>
    );
  }

  if (!user) {
    return (
      <AuthScreen
        request={api}
        onLogin={login}
      />
    );
  }

  return (
    <View
      style={[
        s.root,
        {
          flexDirection:
            width > 1050 ? 'row' : 'column',
        },
      ]}
    >
      <View
        style={[
          s.sidebar,
          width <= 1050 && {
            width: '100%',
            padding: 14,
          },
        ]}
      >
        <Text style={s.brand}>
          rent
          <Text style={{ color: '#71e5b8' }}>
            control.
          </Text>
        </Text>

        <ScrollView
          horizontal={width <= 1050}
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={{ gap: 5 }}
        >
          {menu
            .filter(
              (name) =>
                name !== 'Usuarios' || admin,
            )
            .map((name) => (
              <Pressable
                key={name}
                onPress={() => navigate(name)}
                style={[
                  s.nav,
                  page === name && {
                    backgroundColor: '#235e48',
                  },
                ]}
              >
                <Text
                  style={{
                    color: '#fff',
                    fontWeight:
                      page === name ? '700' : '400',
                  }}
                >
                  {name}
                </Text>
              </Pressable>
            ))}

          <Pressable
            onPress={logout}
            style={s.nav}
          >
            <Text style={{ color: '#f1c2af' }}>
              Cerrar sesión
            </Text>
          </Pressable>
        </ScrollView>

        {width > 1050 && (
          <View style={{ paddingTop: 20 }}>
            <Text
              style={{
                color: '#fff',
                fontWeight: '700',
              }}
            >
              {user.name}
            </Text>

            <Text
              style={{
                color: '#a4c7b4',
                marginTop: 7,
              }}
            >
              {admin
                ? 'Administrador'
                : 'Usuario inquilino'}
            </Text>
          </View>
        )}
      </View>

      <ScrollView
        style={{ flex: 1 }}
        contentContainerStyle={{
          padding: width > 1050 ? 32 : 18,
          paddingBottom: 60,
          maxWidth: 1400,
          width: '100%',
          alignSelf: 'center',
        }}
        keyboardShouldPersistTaps="handled"
      >
        <View style={s.between}>
          <Text style={s.muted}>
            Mi espacio / {page}
          </Text>

          <Button
            title="Actualizar"
            secondary
            onPress={refresh}
          />
        </View>

        {!!error && (
          <View style={s.error}>
            <Text
              accessibilityRole="alert"
              style={{ color: '#9b352e' }}
            >
              {error}
            </Text>
          </View>
        )}

        {!!notice && (
          <View style={s.success}>
            <Text style={{ color: '#1b684d' }}>
              {notice}
            </Text>
          </View>
        )}

        {data?.mode === 'demo' && (
          <Text style={s.small}>
            Almacenamiento local · Datos de ejemplo
            y tus registros
          </Text>
        )}

        {!data && (
          <ActivityIndicator color="#218761" />
        )}

        {page === 'Inicio' && metrics && (
          <>
            <View style={s.hero}>
              <Text
                style={{
                  color: '#8de2bc',
                  fontSize: 12,
                  letterSpacing: 2,
                }}
              >
                TU ESPACIO. TU CONTROL.
              </Text>

              <Text style={s.heroTitle}>
                {admin
                  ? 'Todo bajo control.'
                  : 'Tu renta, al día.'}
              </Text>

              <Text
                style={{
                  color: '#c4ddcf',
                  lineHeight: 24,
                }}
              >
                {admin
                  ? 'Administra tus inmuebles y atiende tus pendientes desde un solo lugar.'
                  : 'Consulta tu contrato, pagos y mantenimientos.'}
              </Text>

              {admin && (
                <Button
                  title="Nueva propiedad"
                  onPress={() => open('property')}
                />
              )}
            </View>

            {!admin && !properties.length && (
              <Card>
                <Text style={s.h2}>
                  Tu cuenta ya está lista
                </Text>

                <Text style={s.body}>
                  El administrador debe asignarte
                  tu inmueble desde Usuarios.
                  Después aparecerán aquí tu
                  contrato y pagos.
                </Text>
              </Card>
            )}

            <View style={s.grid}>
              {[
                [
                  'Cobrado',
                  money(metrics.collected_cents),
                ],
                [
                  'Saldo vencido',
                  money(metrics.overdue_cents),
                ],
                [
                  'Ocupación',
                  `${metrics.occupied} / ${metrics.properties}`,
                ],
                [
                  'Alertas',
                  String(data.alerts.length),
                ],
              ].map(([label, value]) => (
                <View
                  key={label}
                  style={[
                    s.card,
                    {
                      flexGrow: 1,
                      minWidth: 170,
                    },
                  ]}
                >
                  <Text style={s.muted}>
                    {label}
                  </Text>

                  <Text style={s.stat}>
                    {value}
                  </Text>
                </View>
              ))}
            </View>

            <Text style={s.small}>
              Importes acumulados, no únicamente
              del mes.
            </Text>

            <View style={s.grid}>
              <View
                style={{
                  flex: 1,
                  minWidth: 260,
                }}
              >
                <Card>
                  <Text style={s.h2}>
                    Cobrado y pendiente
                  </Text>

                  <PieChart
                    rows={[
                      {
                        label: 'Cobrado',
                        value:
                          metrics.collected_cents,
                        display: money(
                          metrics.collected_cents,
                        ),
                        color: '#218761',
                      },
                      {
                        label: 'Pendiente',
                        value:
                          metrics.charged_cents -
                          metrics.collected_cents,
                        display: money(
                          metrics.charged_cents -
                            metrics.collected_cents,
                        ),
                        color: '#e8ad58',
                      },
                    ]}
                  />
                </Card>
              </View>

              <View
                style={{
                  flex: 1,
                  minWidth: 260,
                }}
              >
                <Card>
                  <Text style={s.h2}>
                    Ocupación de inmuebles
                  </Text>

                  <PieChart
                    rows={[
                      {
                        label: 'Ocupados',
                        value: metrics.occupied,
                        color: '#218761',
                      },
                      {
                        label: 'Disponibles',
                        value:
                          metrics.properties -
                          metrics.occupied,
                        color: '#b8dcca',
                      },
                    ]}
                  />
                </Card>
              </View>
            </View>

            <Text style={s.h2}>
              Atención prioritaria
            </Text>

            {data.alerts
              .slice(0, 3)
              .map((a: any) => (
                <Card key={a.id}>
                  <Badge
                    text={
                      a.kind === 'mantenimiento'
                        ? 'Mantenimiento urgente'
                        : 'Cobranza'
                    }
                    warn
                  />

                  <Text style={s.h2}>
                    {a.property}
                  </Text>

                  <Text style={s.body}>
                    {a.reason}
                  </Text>

                  <Button
                    title="Ver inmueble"
                    secondary
                    onPress={() => {
                      setSelected(a.property_id);
                      navigate('Propiedades');
                    }}
                  />
                </Card>
              ))}

            {!data.alerts.length && (
              <Card>
                <Text style={s.body}>
                  No tienes alertas pendientes.
                </Text>
              </Card>
            )}
          </>
        )}

        {page === 'Propiedades' && (
          <>
            {heading(
              admin
                ? 'Tus propiedades'
                : 'Mi inmueble',
              'Abre el expediente para consultar contrato, cargos y mantenimientos.',
            )}

            <View style={s.between}>
              <TextInput
                accessibilityLabel="Buscar propiedad"
                value={query}
                onChangeText={setQuery}
                placeholder="Buscar por nombre o dirección"
                style={[
                  s.input,
                  {
                    flex: 1,
                    minWidth: 200,
                  },
                ]}
              />

              {admin && (
                <Button
                  title="Registrar propiedad"
                  onPress={() => open('property')}
                />
              )}
            </View>

            {properties
              .filter((p: any) =>
                (
                  p.name +
                  ' ' +
                  (p.address || '')
                )
                  .toLowerCase()
                  .includes(query.toLowerCase()),
              )
              .map((p: any) => (
                <Card key={p.id}>
                  <View style={s.between}>
                    <Text style={s.h2}>
                      {p.name}
                    </Text>

                    <Badge
                      text={
                        p.contract_active
                          ? 'Contrato activo'
                          : 'Disponible'
                      }
                    />
                  </View>

                  <Text style={s.muted}>
                    {p.address ||
                      'Dirección no capturada en este ejemplo'}
                  </Text>

                  <Text style={s.body}>
                    {p.monthly_rent_cents
                      ? money(
                          p.monthly_rent_cents,
                        ) + ' / mes'
                      : 'Renta mensual sin capturar'}
                  </Text>

                  <Button
                    title={
                      selected === p.id
                        ? 'Cerrar expediente'
                        : 'Abrir expediente'
                    }
                    secondary
                    onPress={() =>
                      setSelected(
                        selected === p.id
                          ? ''
                          : p.id,
                      )
                    }
                  />

                  {selected === p.id && (
                    <>
                      <View style={s.grid}>
                        {admin &&
                          !p.contract_active && (
                            <Button
                              title="Asignar contrato"
                              onPress={() =>
                                open('contract', p)
                              }
                            />
                          )}

                        {admin &&
                          p.contract_active && (
                            <Button
                              title="Generar cargo"
                              onPress={() =>
                                open('charge', p)
                              }
                            />
                          )}

                        <Button
                          title="Reportar mantenimiento"
                          secondary
                          onPress={() =>
                            open('maintenance', p)
                          }
                        />

                        {!!p.address && (
                          <Button
                            title="Ver ubicación"
                            secondary
                            onPress={() =>
                              Linking.openURL(
                                'https://www.google.com/maps/search/?api=1&query=' +
                                  encodeURIComponent(
                                    p.address,
                                  ),
                              )
                            }
                          />
                        )}
                      </View>

                      <Text style={s.h2}>
                        Contrato
                      </Text>

                      <Text style={s.body}>
                        {p.contract?.tenant ||
                          (p.contract_active
                            ? 'Contrato de demostración sin nombre capturado'
                            : 'Sin contrato activo')}
                      </Text>

                      {p.contract && (
                        <Text style={s.muted}>
                          {p.contract.start} al{' '}
                          {p.contract.end}
                        </Text>
                      )}

                      <Text style={s.h2}>
                        Cargos
                      </Text>

                      {p.charges.map((c: any) =>
                        chargeCard({
                          ...c,
                          property: p,
                        }),
                      )}

                      {!p.charges.length && (
                        <Text style={s.muted}>
                          No hay cargos registrados.
                        </Text>
                      )}

                      <Text style={s.h2}>
                        Mantenimientos
                      </Text>

                      {p.maintenance.map(
                        (m: any) => (
                          <View
                            key={m.id}
                            style={s.line}
                          >
                            <Text style={s.body}>
                              {m.description}
                            </Text>

                            <Badge
                              text={m.status}
                              warn={
                                m.status !==
                                'resuelto'
                              }
                            />
                          </View>
                        ),
                      )}
                    </>
                  )}
                </Card>
              ))}

            {!properties.length && (
              <Card>
                <Text style={s.body}>
                  {admin
                    ? 'Registra tu primer inmueble.'
                    : 'Aún no tienes un inmueble asignado.'}
                </Text>
              </Card>
            )}
          </>
        )}

        {page === 'Contratos' && (
          <>
            {heading(
              admin
                ? 'Contratos e inquilinos'
                : 'Mis contratos',
              'Consulta el inquilino y la vigencia registrada.',
            )}

            {properties.map((p: any) => (
              <Card key={p.id}>
                <Text style={s.h2}>{p.name}</Text>

                <Badge
                  text={
                    p.contract_active
                      ? 'Activo'
                      : 'Sin contrato'
                  }
                />

                <Text style={s.body}>
                  {p.contract?.tenant ||
                    'Sin nombre de inquilino capturado'}
                </Text>

                {p.contract && (
                  <Text style={s.muted}>
                    {p.contract.start} →{' '}
                    {p.contract.end}
                  </Text>
                )}

                {admin && !p.contract_active && (
                  <Button
                    title="Registrar contrato"
                    onPress={() =>
                      open('contract', p)
                    }
                  />
                )}

                <Button
                  title="Ver expediente"
                  secondary
                  onPress={() => showProperty(p)}
                />
              </Card>
            ))}
          </>
        )}

        {page === 'Cobranza' && (
          <>
            {heading(
              admin
                ? 'Cobranza'
                : 'Mis cargos y pagos',
              admin
                ? 'Registra pagos o abonos y consulta el saldo.'
                : 'Consulta los cargos y pagos registrados por la administración.',
            )}

            <View style={s.grid}>
              {[
                'Todos',
                'Pendientes',
                'Liquidados',
              ].map((f) => (
                <Button
                  key={f}
                  title={f}
                  secondary={filter !== f}
                  onPress={() => setFilter(f)}
                />
              ))}
            </View>

            {charges
              .filter(
                (c: any) =>
                  filter === 'Todos' ||
                  (filter === 'Pendientes'
                    ? c.amount_cents >
                      c.paid_cents
                    : c.amount_cents ===
                      c.paid_cents),
              )
              .map(chargeCard)}

            {admin && (
              <Button
                title="Generar cargo desde Propiedades"
                secondary
                onPress={() =>
                  navigate('Propiedades')
                }
              />
            )}
          </>
        )}

        {page === 'Mantenimientos' && (
          <>
            {heading(
              'Mantenimientos',
              'Reporta problemas y consulta su seguimiento.',
            )}

            <Button
              title="Reportar desde mi inmueble"
              secondary
              onPress={() =>
                navigate('Propiedades')
              }
            />

            <View style={s.grid}>
              {[
                'Todos',
                'Abiertos',
                'Resueltos',
              ].map((f) => (
                <Button
                  key={f}
                  title={f}
                  secondary={filter !== f}
                  onPress={() => setFilter(f)}
                />
              ))}
            </View>

            {tasks
              .filter(
                (m: any) =>
                  filter === 'Todos' ||
                  (filter === 'Abiertos'
                    ? m.status !== 'resuelto'
                    : m.status === 'resuelto'),
              )
              .map((m: any) => (
                <Card key={m.id}>
                  <Text style={s.h2}>
                    {m.property.name}
                  </Text>

                  <Badge
                    text={
                      m.status === 'resuelto'
                        ? 'Resuelto'
                        : m.urgent
                          ? 'Urgente'
                          : 'Pendiente'
                    }
                    warn={
                      m.status !== 'resuelto'
                    }
                  />

                  <Text style={s.body}>
                    {m.description}
                  </Text>

                  {admin &&
                    m.status !== 'resuelto' && (
                      <Button
                        disabled={busy}
                        title="Marcar resuelto"
                        onPress={() => resolve(m)}
                      />
                    )}
                </Card>
              ))}
          </>
        )}

        {page === 'Alertas' && data && (
          <>
            {heading(
              'Centro de alertas',
              'Consulta el motivo y abre el inmueble para atenderlo.',
            )}

            {data.alerts.map((a: any) => (
              <Card key={a.id}>
                <Badge
                  warn
                  text={
                    a.priority === 'critica'
                      ? 'Atención urgente'
                      : 'Prioridad alta'
                  }
                />

                <Text style={s.h2}>
                  {a.property}
                </Text>

                <Text style={s.body}>
                  {a.reason}
                </Text>

                <View style={s.grid}>
                  <Button
                    title="Ver inmueble"
                    onPress={() => {
                      setSelected(a.property_id);
                      navigate('Propiedades');
                    }}
                  />

                  <Button
                    title="Entender el cálculo"
                    secondary
                    onPress={() =>
                      navigate('Cómo se calculó')
                    }
                  />
                </View>
              </Card>
            ))}

            {!data.alerts.length && (
              <Card>
                <Text style={s.body}>
                  Todo al día. No hay alertas
                  pendientes.
                </Text>
              </Card>
            )}
          </>
        )}

        {page === 'Movimientos' && (
          <>
            {heading(
              'Historial de pagos',
              'Cada abono nuevo conserva fecha, importe y referencia.',
            )}

            {payments.map((p: any) => (
              <Card key={p.id}>
                <View style={s.between}>
                  <Text style={s.h2}>
                    {p.property.name}
                  </Text>

                  <Text
                    style={[
                      s.h2,
                      { color: '#218761' },
                    ]}
                  >
                    + {money(p.amount_cents)}
                  </Text>
                </View>

                <Text style={s.body}>
                  {p.date} ·{' '}
                  {p.reference || 'Sin referencia'}
                </Text>

                <Text style={s.small}>
                  Folio: {p.id}
                </Text>
              </Card>
            ))}

            {!payments.length && (
              <Card>
                <Text style={s.body}>
                  Aún no hay movimientos nuevos.
                  Los pagos de los ejemplos
                  iniciales son saldos de apertura
                  y no tienen movimientos
                  individuales.
                </Text>
              </Card>
            )}
          </>
        )}

        {page === 'RentIA' && (
          <>
            {heading(
              'Tu asistente RentIA',
              'Consulta alertas y totales mediante Ollama.',
            )}

            <Card>
              <Text style={s.h2}>
                ¿En qué puedo ayudarte?
              </Text>

              <Text style={s.muted}>
                Python calcula y RentIA explica.
                Requiere Ollama activo y un
                modelo compatible.
              </Text>

              <TextInput
                accessibilityLabel="Pregunta a RentIA"
                multiline
                value={question}
                onChangeText={setQuestion}
                style={[
                  s.input,
                  {
                    minHeight: 110,
                    marginVertical: 15,
                  },
                ]}
              />

              <Button
                disabled={
                  busy || !question.trim()
                }
                title={
                  busy
                    ? 'Consultando…'
                    : 'Consultar asistente'
                }
                onPress={ask}
              />

              {!!answer && (
                <Text selectable style={s.body}>
                  {answer}
                </Text>
              )}
            </Card>
          </>
        )}

        {page === 'Usuarios' && admin && (
          <>
            {heading(
              'Usuarios e inmuebles',
              'Asigna a cada inquilino la información que puede consultar.',
            )}

            <Card>
              <Text style={s.body}>
                El inquilino crea su cuenta desde
                el login. Después selecciona su
                cuenta al asignar el inmueble.
                Esta asignación de acceso no
                modifica el contrato.
              </Text>
            </Card>

            {users.map((u) => (
              <Card key={u.id}>
                <Text style={s.h2}>
                  {u.name}
                </Text>

                <Text style={s.muted}>
                  {u.email}
                </Text>

                <Text style={s.body}>
                  Inmuebles:{' '}
                  {properties
                    .filter(
                      (p: any) =>
                        p.assigned_user_id ===
                        u.id,
                    )
                    .map((p: any) => p.name)
                    .join(', ') || 'Ninguno'}
                </Text>
              </Card>
            ))}

            {!users.length && (
              <Text style={s.body}>
                Todavía no hay cuentas de usuario.
              </Text>
            )}

            <Text style={s.h2}>
              Asignaciones por inmueble
            </Text>

            {properties.map((p: any) => (
              <Card key={p.id}>
                <Text style={s.h2}>
                  {p.name}
                </Text>

                <Text style={s.muted}>
                  {users.find(
                    (u) =>
                      u.id ===
                      p.assigned_user_id,
                  )?.email ||
                    'Sin usuario asignado'}
                </Text>

                <Button
                  title="Asignar o cambiar usuario"
                  onPress={() => open('access', p)}
                />
              </Card>
            ))}
          </>
        )}

        {page === 'Cómo se calculó' && (
          <>
            {heading(
              'Entiende tus resultados',
              'Explicaciones con los datos de tus inmuebles.',
            )}

            <View style={s.grid}>
              {[
                'Alertas',
                'Saldos',
                'Gráficas',
                'Recorridos',
              ].map((t) => (
                <Button
                  key={t}
                  title={t}
                  secondary={calculation !== t}
                  onPress={() =>
                    setCalculation(t)
                  }
                />
              ))}
            </View>

            <Button
              title="Actualizar cálculos"
              secondary
              onPress={async () => {
                try {
                  setExplanation(
                    await api('/explanations'),
                  );
                } catch (e: any) {
                  setError(e.message);
                }
              }}
            />

            {!explanation && (
              <Text style={s.muted}>
                Pulsa Actualizar cálculos si la
                explicación todavía no aparece.
              </Text>
            )}

            {explanation &&
              calculation === 'Saldos' && (
                <Card>
                  <Text style={s.h2}>
                    El saldo conserva cada abono
                  </Text>

                  <Text style={s.body}>
                    Saldo = importe del cargo −
                    pagos registrados.
                  </Text>

                  <Text style={s.body}>
                    {money(
                      explanation.totals
                        .charged_cents,
                    )}{' '}
                    −{' '}
                    {money(
                      explanation.totals
                        .collected_cents,
                    )}{' '}
                    ={' '}
                    {money(
                      explanation.totals
                        .pending_cents,
                    )}{' '}
                    pendientes.
                  </Text>

                  <Text style={s.muted}>
                    Vencido:{' '}
                    {money(
                      explanation.totals
                        .overdue_cents,
                    )}
                    . Solo incluye cargos
                    anteriores a{' '}
                    {explanation.as_of} con saldo.
                    Un cargo que vence hoy todavía
                    no está vencido.
                  </Text>

                  {explanation.properties.map(
                    (p: any) =>
                      p.charges.map((c: any) => (
                        <View
                          key={c.id}
                          style={s.line}
                        >
                          <Text style={s.h2}>
                            {p.name}
                          </Text>

                          <Text style={s.body}>
                            {money(c.amount_cents)}{' '}
                            −{' '}
                            {money(c.paid_cents)}{' '}
                            ={' '}
                            {money(c.balance_cents)}
                          </Text>

                          <Text style={s.muted}>
                            Vence: {c.due} ·{' '}
                            {c.balance_cents === 0
                              ? 'Liquidado'
                              : c.Q
                                ? 'Vencido hace ' +
                                  c.days_overdue +
                                  ' días'
                                : 'Dentro del plazo'}
                          </Text>
                        </View>
                      )),
                  )}
                </Card>
              )}

            {explanation &&
              calculation === 'Gráficas' && (
                <>
                  <Card>
                    <Text style={s.h2}>
                      Porcentaje cobrado
                    </Text>

                    <Text style={s.body}>
                      Cobrado ÷ total de cargos ×
                      100.
                    </Text>

                    <Text style={s.body}>
                      {explanation.totals
                        .charged_cents
                        ? `${money(explanation.totals.collected_cents)} ÷ ${money(explanation.totals.charged_cents)} × 100 = ${explanation.totals.collected_percent}%`
                        : 'No hay cargos: mostramos 0% y una gráfica sin datos.'}
                    </Text>

                    <Text style={s.muted}>
                      Son importes acumulados, no
                      un reporte mensual.
                    </Text>
                  </Card>

                  <Card>
                    <Text style={s.h2}>
                      Ocupación
                    </Text>

                    <Text style={s.body}>
                      Inmuebles con contrato activo
                      ÷ total de inmuebles × 100.
                    </Text>

                    <Text style={s.body}>
                      {explanation.totals
                        .properties
                        ? `${explanation.totals.occupied} ÷ ${explanation.totals.properties} × 100 = ${explanation.totals.occupancy_percent}%`
                        : 'No hay inmuebles: mostramos 0%.'}
                    </Text>
                  </Card>
                </>
              )}

            {explanation &&
              calculation === 'Alertas' && (
                <>
                  <Card>
                    <Text style={s.h2}>
                      Cuatro preguntas para evaluar
                      una alerta
                    </Text>

                    {[
                      'P · ¿Existe contrato activo?',
                      'Q · ¿Ya pasó la fecha límite de este cargo?',
                      'R · ¿Ese mismo cargo conserva saldo?',
                      'S · ¿Existe mantenimiento urgente sin resolver?',
                    ].map((t) => (
                      <Text
                        key={t}
                        style={s.body}
                      >
                        {t}
                      </Text>
                    ))}

                    <Text style={s.body}>
                      La cobranza de contrato
                      activo requiere P, Q y R
                      verdaderas. El mantenimiento
                      urgente genera su propia
                      alerta aunque el inmueble
                      esté disponible.
                    </Text>

                    <Text style={s.muted}>
                      La deuda de un contrato
                      terminado también se
                      conserva: se evalúa con
                      NO P, Q y R.
                    </Text>
                  </Card>

                  {explanation.properties.map(
                    (p: any) => (
                      <Card key={p.id}>
                        <Text style={s.h2}>
                          {p.name}
                        </Text>

                        <Text style={s.body}>
                          Contrato activo:{' '}
                          {p.P ? 'Sí' : 'No'} ·
                          Urgencia abierta:{' '}
                          {p.S ? 'Sí' : 'No'}
                        </Text>

                        {p.charges.map(
                          (c: any) => (
                            <View
                              key={c.id}
                              style={s.line}
                            >
                              <Text style={s.body}>
                                Vence {c.due} ·
                                Saldo{' '}
                                {money(
                                  c.balance_cents,
                                )}
                              </Text>

                              <Text style={s.muted}>
                                P:{' '}
                                {c.P ? 'Sí' : 'No'}
                                {' · '}Q:{' '}
                                {c.Q ? 'Sí' : 'No'}
                                {' · '}R:{' '}
                                {c.R ? 'Sí' : 'No'}
                                {' · '}S:{' '}
                                {c.S ? 'Sí' : 'No'}
                              </Text>

                              <Text style={s.body}>
                                {c.collection
                                  ? 'Se genera alerta de renta vencida.'
                                  : c.historical
                                    ? 'Se genera alerta de deuda anterior.'
                                    : 'Este cargo no genera alerta de cobranza.'}
                              </Text>
                            </View>
                          ),
                        )}

                        {p.S && (
                          <Text style={s.body}>
                            Se genera además una
                            alerta de mantenimiento
                            urgente.
                          </Text>
                        )}

                        {!p.charges.length && (
                          <Text style={s.muted}>
                            Sin cargos registrados.
                          </Text>
                        )}
                      </Card>
                    ),
                  )}

                  <Card>
                    <Text style={s.h2}>
                      Reglas y conectores
                    </Text>

                    <Text style={s.body}>
                      Y (∧): ambas condiciones.
                      O (∨): una o ambas.
                      NO (¬): la condición no se
                      cumple. Si… entonces (→):
                      una condición implica un
                      resultado. Si y solo si (↔):
                      el resultado corresponde
                      exactamente a su condición.
                    </Text>

                    <Text style={s.body}>
                      Contrato activo:
                      A = (P ∧ Q ∧ R) ∨ S.
                    </Text>

                    <Text style={s.body}>
                      Deuda anterior:
                      ¬P ∧ Q ∧ R. Esta es una
                      regla adicional.
                    </Text>

                    <Button
                      title={
                        showTable
                          ? 'Ocultar tabla'
                          : 'Ver las 16 combinaciones PQRS'
                      }
                      secondary
                      onPress={() =>
                        setShowTable(!showTable)
                      }
                    />

                    {showTable &&
                      explanation.table.map(
                        (r: any, i: number) => (
                          <Text
                            key={i}
                            style={s.muted}
                          >
                            {[
                              'P',
                              'Q',
                              'R',
                              'S',
                              'A',
                            ]
                              .map(
                                (k) =>
                                  `${k}: ${r[k] ? 'V' : 'F'}`,
                              )
                              .join('   ')}
                          </Text>
                        ),
                      )}
                  </Card>
                </>
              )}

            {calculation === 'Recorridos' && (
              <>
                {[
                  [
                    'BFS sin estimaciones',
                    'Revisa posiciones por niveles. Cada movimiento cuesta un paso; encuentra el camino con menos pasos si existe.',
                  ],
                  [
                    'A* y Manhattan',
                    'A* usa f = g + h. g cuenta los pasos recorridos y h estima lo restante: |fila actual − fila meta| + |columna actual − columna meta|. Se aplica a la cuadrícula de cuatro direcciones, no directamente a coordenadas GPS.',
                  ],
                  [
                    'Hill Climbing',
                    'Intercambia dos visitas y conserva el cambio si reduce la distancia. Se detiene cuando no encuentra mejora; puede quedar en un óptimo local.',
                  ],
                  [
                    'Algoritmo genético',
                    'Genera órdenes de visitas, selecciona candidatos, cruza recorridos sin repetir destinos y realiza mutaciones. Conserva los mejores y usa semilla 42 para repetir el ejemplo. No garantiza el óptimo.',
                  ],
                  [
                    'Marco PEAS',
                    'Desempeño: alertas correctas y recorridos válidos. Entorno: inmuebles, cargos, contratos y mapa. Acciones: presentar alertas y proponer recorridos. Sensores: datos guardados, fechas y ubicaciones. Python calcula y Ollama explica.',
                  ],
                ].map(([title, text]) => (
                  <Card key={title}>
                    <Text style={s.h2}>
                      {title}
                    </Text>

                    <Text style={s.body}>
                      {text}
                    </Text>
                  </Card>
                ))}

                <Button
                  title="Ver resultados en el laboratorio"
                  onPress={() =>
                    navigate('Laboratorio')
                  }
                />
              </>
            )}
          </>
        )}

        {page === 'Ayuda' && (
          <>
            {heading(
              'Estamos para ayudarte',
              'Guía, contactos e información de RentControl.',
            )}

            <View style={s.grid}>
              {[
                'Guía rápida',
                'Preguntas frecuentes',
                'Contactos',
                'Acerca de',
              ].map((t) => (
                <Button
                  key={t}
                  title={t}
                  secondary={help !== t}
                  onPress={() => setHelp(t)}
                />
              ))}
            </View>

            {help === 'Guía rápida' && (
              <Card>
                <Text style={s.h2}>
                  {admin
                    ? 'Primeros pasos de administración'
                    : 'Cómo utilizar tu cuenta'}
                </Text>

                <Text style={s.body}>
                  {admin
                    ? '1. Registra un inmueble.\n2. Asigna contrato e inquilino.\n3. Genera el cargo.\n4. Registra pagos o abonos.\n5. Asigna acceso al inquilino desde Usuarios.'
                    : '1. Solicita la asignación de tu inmueble.\n2. Consulta contrato y pagos.\n3. Reporta problemas desde el expediente.\n4. Revisa alertas y Cómo se calculó.'}
                </Text>
              </Card>
            )}

            {help === 'Preguntas frecuentes' && (
              <>
                {[
                  [
                    '¿Por qué hay saldo vencido?',
                    'La fecha del cargo pasó y queda saldo. Al liquidarlo desaparece la alerta.',
                  ],
                  [
                    '¿Por qué no veo inmuebles?',
                    'El administrador debe asignarte el inmueble desde Usuarios.',
                  ],
                  [
                    '¿Por qué RentIA no responde?',
                    'Ollama debe estar activo con un modelo compatible. Los formularios funcionan sin el asistente.',
                  ],
                  [
                    '¿Olvidaste la contraseña?',
                    'La recuperación aún no está implementada. Contacta al responsable de instalar el sistema; no borres la base de datos.',
                  ],
                ].map(([title, text]) => (
                  <Card key={title}>
                    <Text style={s.h2}>
                      {title}
                    </Text>

                    <Text style={s.body}>
                      {text}
                    </Text>
                  </Card>
                ))}
              </>
            )}

            {help === 'Contactos' && (
              <Card>
                <Text style={s.h2}>
                  Contacto de tu cuenta
                </Text>

                <Text style={s.body}>
                  {user.name} · {user.email}
                </Text>

                <Button
                  title="Abrir correo"
                  secondary
                  onPress={() =>
                    Linking.openURL(
                      'mailto:' +
                        encodeURIComponent(
                          user.email,
                        ),
                    )
                  }
                />

                <Text style={s.muted}>
                  El teléfono y correo de soporte
                  técnico están pendientes de
                  configurar.
                </Text>
              </Card>
            )}

            {help === 'Acerca de' && (
              <Card>
                <Text style={s.h2}>
                  RentControl
                </Text>

                <Text style={s.body}>
                  Administración de propiedades,
                  contratos, cobros y
                  mantenimientos con cuentas de
                  administrador e inquilino.
                </Text>

                <Text style={s.muted}>
                  React Native, Expo y Python.
                  Datos locales en SQLite.
                  La conexión definitiva con
                  MongoDB queda pendiente.
                </Text>
              </Card>
            )}
          </>
        )}

        {page === 'Laboratorio' && (
          <>
            {heading(
              'Laboratorio académico',
              'Mapa de demostración. No representa rutas viales GPS.',
            )}

            <Button
              title={
                busy
                  ? 'Calculando…'
                  : 'Calcular demostración'
              }
              disabled={busy}
              onPress={loadLab}
            />

            {lab && (
              <>
                <Card>
                  <Text style={s.h2}>
                    Mapa de cuatro direcciones
                  </Text>

                  <ScrollView horizontal>
                    <View>
                      {lab.grid.map(
                        (
                          row: number[],
                          i: number,
                        ) => (
                          <View
                            key={i}
                            style={{
                              flexDirection: 'row',
                            }}
                          >
                            {row.map(
                              (value, j) => {
                                const point =
                                  lab.points.findIndex(
                                    (p: number[]) =>
                                      p[0] === i &&
                                      p[1] === j,
                                  );

                                return (
                                  <View
                                    key={j}
                                    style={{
                                      width: 34,
                                      height: 34,
                                      margin: 2,
                                      borderRadius: 5,
                                      backgroundColor:
                                        value
                                          ? '#234d3b'
                                          : '#dcefe3',
                                      alignItems:
                                        'center',
                                      justifyContent:
                                        'center',
                                    }}
                                  >
                                    <Text>
                                      {point < 0
                                        ? ''
                                        : point}
                                    </Text>
                                  </View>
                                );
                              },
                            )}
                          </View>
                        ),
                      )}
                    </View>
                  </ScrollView>

                  <Text style={s.muted}>
                    0: oficina · 1–4: visitas ·
                    Oscuro: obstáculo
                  </Text>

                  <Text style={s.body}>
                    Manhattan inicial:
                    |{lab.points[0][0]} −{' '}
                    {lab.points[2][0]}| + |
                    {lab.points[0][1]} −{' '}
                    {lab.points[2][1]}| ={' '}
                    {Math.abs(
                      lab.points[0][0] -
                        lab.points[2][0],
                    ) +
                      Math.abs(
                        lab.points[0][1] -
                          lab.points[2][1],
                      )}{' '}
                    pasos estimados.
                  </Text>

                  {lab.searches.map((r: any) => (
                    <View
                      key={r.method}
                      style={s.line}
                    >
                      <Text style={s.h2}>
                        {r.method === 'bfs'
                          ? 'BFS'
                          : 'A* con Manhattan'}
                      </Text>

                      <Text style={s.body}>
                        Costo:{' '}
                        {r.cost ?? 'Sin camino'}{' '}
                        pasos · Nodos explorados:{' '}
                        {r.expanded}
                      </Text>

                      <Text style={s.small}>
                        Recorrido:{' '}
                        {r.path
                          .map(
                            (p: number[]) =>
                              '(' +
                              p.join(',') +
                              ')',
                          )
                          .join(' → ')}
                      </Text>
                    </View>
                  ))}
                </Card>

                {lab.routes.map((r: any) => (
                  <Card key={r.method}>
                    <Text style={s.h2}>
                      {r.method === 'hill_climbing'
                        ? 'Hill Climbing'
                        : r.method === 'genetic'
                          ? 'Algoritmo genético'
                          : 'Óptimo por enumeración'}
                    </Text>

                    <Text style={s.body}>
                      {r.order.join(' → ')} ·{' '}
                      {r.cost} pasos
                    </Text>

                    {r.initial !== undefined && (
                      <Text style={s.muted}>
                        Inicial: {r.initial} ·
                        Reducción:{' '}
                        {r.initial - r.cost} pasos.
                      </Text>
                    )}

                    <Text style={s.muted}>
                      Sumamos distancias BFS entre
                      visitas, incluyendo el
                      regreso a la oficina.
                    </Text>
                  </Card>
                ))}

                <Card>
                  <Text style={s.h2}>
                    Tabla de verdad PQRS
                  </Text>

                  <Text style={s.body}>
                    A = (P ∧ Q ∧ R) ∨ S
                  </Text>

                  {lab.logic.table.map(
                    (r: any, i: number) => (
                      <Text
                        key={i}
                        style={s.muted}
                      >
                        {['P', 'Q', 'R', 'S', 'A']
                          .map(
                            (k) =>
                              k +
                              ': ' +
                              (r[k] ? 'V' : 'F'),
                          )
                          .join('   ')}
                      </Text>
                    ),
                  )}
                </Card>
              </>
            )}
          </>
        )}
      </ScrollView>

      {form && (
        <View style={s.overlay}>
          <ScrollView
            contentContainerStyle={{
              flexGrow: 1,
              alignItems: 'center',
              justifyContent: 'center',
              padding: 20,
            }}
            keyboardShouldPersistTaps="handled"
          >
            <View style={s.modal}>
              <Text style={s.h2}>
                {
                  (
                    {
                      property: 'Nueva propiedad',
                      contract: 'Registrar contrato',
                      charge: 'Generar cargo',
                      payment: 'Registrar abono',
                      maintenance:
                        'Reportar mantenimiento',
                      access: 'Asignar acceso',
                    } as Record<string, string>
                  )[form.type]
                }
              </Text>

              <Text style={s.muted}>
                {form.property?.name || ''}
              </Text>

              {form.type === 'property' && (
                <>
                  <Field
                    label="Nombre del inmueble"
                    value={values.name}
                    onChange={field('name')}
                  />

                  <Field
                    label="Dirección completa"
                    value={values.address}
                    onChange={field('address')}
                  />
                </>
              )}

              {[
                'property',
                'contract',
                'charge',
              ].includes(form.type) && (
                <Field
                  label={
                    form.type === 'charge'
                      ? 'Importe del cargo en MXN'
                      : 'Renta mensual en MXN'
                  }
                  numeric
                  value={values.rent}
                  onChange={field('rent')}
                />
              )}

              {form.type === 'contract' && (
                <>
                  <Field
                    label="Nombre del inquilino"
                    value={values.tenant}
                    onChange={field('tenant')}
                  />

                  <Field
                    label="Inicio AAAA-MM-DD"
                    value={values.start}
                    onChange={field('start')}
                  />

                  <Field
                    label="Fin AAAA-MM-DD"
                    value={values.end}
                    onChange={field('end')}
                  />

                  <Text style={s.muted}>
                    Las fechas deben incluir hoy
                    para registrar un contrato
                    vigente.
                  </Text>
                </>
              )}

              {form.type === 'charge' && (
                <Field
                  label="Vencimiento AAAA-MM-DD"
                  value={values.due}
                  onChange={field('due')}
                />
              )}

              {form.type === 'payment' && (
                <>
                  <Text style={s.body}>
                    Saldo:{' '}
                    {money(
                      form.charge.amount_cents -
                        form.charge.paid_cents,
                    )}
                  </Text>

                  <Field
                    label="Importe del abono en MXN"
                    numeric
                    value={values.amount}
                    onChange={field('amount')}
                  />

                  <Field
                    label="Referencia o nota"
                    value={values.reference}
                    onChange={field('reference')}
                  />
                </>
              )}

              {form.type === 'maintenance' && (
                <>
                  <Field
                    label="Describe el problema"
                    value={values.description}
                    onChange={field('description')}
                  />

                  <Button
                    title={
                      values.urgent
                        ? '✓ Urgente'
                        : 'Marcar como urgente'
                    }
                    secondary={!values.urgent}
                    onPress={() =>
                      setValues({
                        ...values,
                        urgent: !values.urgent,
                      })
                    }
                  />
                </>
              )}

              {form.type === 'access' && (
                <>
                  <Text style={s.body}>
                    Selecciona la cuenta que podrá
                    consultar el inmueble.
                    Primero debe registrarse como
                    usuario.
                  </Text>

                  {users.map((u) => (
                    <Button
                      key={u.id}
                      title={
                        u.name + ' · ' + u.email
                      }
                      secondary={
                        values.user_id !== u.id
                      }
                      onPress={() =>
                        setValues({
                          ...values,
                          user_id: u.id,
                        })
                      }
                    />
                  ))}

                  {!users.length && (
                    <Text style={s.muted}>
                      No hay usuarios registrados.
                    </Text>
                  )}
                </>
              )}

              {!!error && (
                <Text
                  accessibilityRole="alert"
                  style={{
                    color: '#ae4036',
                    marginTop: 14,
                  }}
                >
                  {error}
                </Text>
              )}

              <View style={s.grid}>
                <Button
                  title={
                    busy
                      ? 'Guardando…'
                      : 'Guardar registro'
                  }
                  disabled={busy}
                  onPress={save}
                />

                <Button
                  title="Cancelar"
                  secondary
                  disabled={busy}
                  onPress={() => {
                    setForm(null);
                    setError('');
                  }}
                />
              </View>
            </View>
          </ScrollView>
        </View>
      )}
    </View>
  );
}

const s = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: '#f3f6f4',
  },
  sidebar: {
    width: 235,
    backgroundColor: '#123c2f',
    padding: 22,
    paddingTop: 35,
  },
  brand: {
    color: '#fff',
    fontSize: 29,
    fontWeight: '800',
    marginBottom: 22,
    letterSpacing: -1,
  },
  nav: {
    padding: 12,
    borderRadius: 10,
  },
  title: {
    fontSize: 29,
    fontWeight: '800',
    color: '#173e2f',
    marginTop: 18,
    marginBottom: 8,
  },
  h2: {
    fontSize: 18,
    fontWeight: '700',
    color: '#214b37',
    marginVertical: 9,
  },
  body: {
    fontSize: 15,
    lineHeight: 25,
    color: '#3b5b48',
    marginVertical: 9,
  },
  muted: {
    fontSize: 13,
    lineHeight: 22,
    color: '#788e80',
  },
  small: {
    fontSize: 12,
    lineHeight: 20,
    color: '#84978b',
    marginVertical: 8,
  },
  card: {
    backgroundColor: '#fff',
    padding: 22,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: '#e1eae3',
    marginVertical: 8,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginVertical: 12,
  },
  between: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  stat: {
    fontSize: 29,
    fontWeight: '800',
    color: '#194c34',
    marginVertical: 12,
  },
  hero: {
    backgroundColor: '#153f30',
    borderRadius: 22,
    padding: 29,
    marginVertical: 20,
  },
  heroTitle: {
    color: '#fff',
    fontSize: 37,
    fontWeight: '800',
    marginVertical: 14,
    letterSpacing: -1,
  },
  button: {
    backgroundColor: '#218761',
    borderRadius: 10,
    paddingHorizontal: 17,
    paddingVertical: 12,
    marginVertical: 5,
  },
  secondary: {
    backgroundColor: '#e9f3ec',
  },
  badge: {
    backgroundColor: '#e7f4eb',
    color: '#277c54',
    fontSize: 12,
    padding: 8,
    borderRadius: 8,
    alignSelf: 'flex-start',
    overflow: 'hidden',
    marginVertical: 5,
  },
  label: {
    color: '#3b5d49',
    fontSize: 13,
    fontWeight: '600',
    marginTop: 10,
    marginBottom: 7,
  },
  input: {
    borderWidth: 1,
    borderColor: '#dbe7de',
    backgroundColor: '#fafcfb',
    borderRadius: 10,
    padding: 13,
    color: '#264d38',
    fontSize: 15,
  },
  line: {
    borderTopWidth: 1,
    borderTopColor: '#e7eee8',
    paddingTop: 12,
    marginTop: 12,
  },
  error: {
    backgroundColor: '#fff0eb',
    padding: 15,
    borderRadius: 10,
    marginVertical: 12,
  },
  success: {
    backgroundColor: '#e0f1e7',
    padding: 15,
    borderRadius: 10,
    marginVertical: 12,
  },
  overlay: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(7,30,19,.65)',
  },
  modal: {
    backgroundColor: '#fff',
    borderRadius: 22,
    padding: 26,
    width: '100%',
    maxWidth: 530,
  },
});